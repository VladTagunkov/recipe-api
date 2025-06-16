"""
Please provide the full URL to your recipes-api GitHub repository below.
"""
from llama_index.core.agent.legacy.react.base import ReActAgent

from llama_index.llms.openai import OpenAI
import dotenv
import os
from llama_index.core.agent.workflow import FunctionAgent
from github import Github
import asyncio
from llama_index.core.agent.workflow import AgentOutput, ToolCall, ToolCallResult,AgentWorkflow
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context

dotenv.load_dotenv()

context_agent_prompt = """You are the context gathering agent. When gathering context, you MUST gather \n: 
  - The details: author, title, body, diff_url, state, and head_sha; \n
  - Changed files; \n
  - Any requested for files; \n
Once you gather the requested info, you MUST hand control back to the Commentor Agent."""

commenter_agent_system_prompt = """You are the commentor agent that writes review comments for pull requests as a human reviewer would. \n 
Ensure to do the following for a thorough review: 
 - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent. 
 - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing: \n
    - What is good about the PR? \n
    - Did the author follow ALL contribution rules? What is missing? \n
    - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this. \n
    - Are new endpoints documented? - use the diff to determine this. \n 
    - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement. \n
 - If you need any additional details, you must hand off to the Commentor Agent. \n
 - You should directly address the author. So your comments should sound like: \n
 - You must hand off to the ReviewAndPostingAgent once you are done drafting a review. \n
 "Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?"""

review_agent_system_prompt = """You are the Review and Posting agent. You must use the CommentorAgent to create a review comment. 
Once a review is generated, you need to run a final check and post it to GitHub.
   - The review must: \n
   - Be a ~200-300 word review in markdown format. \n
   - Specify what is good about the PR: \n
   - Did the author follow ALL contribution rules? What is missing? \n
   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? \n
   - Are there notes on whether new endpoints were documented? \n
   - Are there suggestions on which lines could be improved upon? Are these lines quoted? \n
 If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. \n
 When you are satisfied, post the review to GitHub.  """


llm = OpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
    system_prompt=context_agent_prompt
)

llm_commenter = OpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
    system_prompt=commenter_agent_system_prompt
)

llm_context = OpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
    system_prompt=context_agent_prompt
)

git_ = Github(os.getenv("GITHUB_TOKEN")) if os.getenv("GITHUB_TOKEN") else Github()
repo_url = os.getenv("REPOSITORY")
pr_number = os.getenv("PR_NUMBER")

def get_sha(repo_):
    """get sha from repo"""
    sha_list=[]
    commits_ = repo_.get_commits()
    for commit_ in commits_:
        sha_list.append(commit_.sha)
    return sha_list


def pr_details_tools(pr_number: str = pr_number):
    """this should return details about the pull request given the number,
        such as the author,
        title,
        body,
        commit SHAs,
        state,
        and more."""
    #git_ = Github(os.getenv("GITHUB_TOKEN")) if os.getenv("GITHUB_TOKEN") else None

    #repo_url_ = "https://github.com/VladTagunkov/recipe-api.git"
    repo_name_ = repo_url.split('/')[-1].replace('.git', '')
    username_ = repo_url.split('/')[-2]
    full_repo_name_ = f"{username_}/{repo_name_}"
    if git_ is not None:
        repo_ = git_.get_repo(full_repo_name_)
        pr = repo_.get_pull(int(pr_number))
        #pr_details_: list[dict[str, any]] = []
        return {
            "The PR author": pr.user.login,
            "title": pr.title,
            "body": pr.body,
            "The diff URL": pr.diff_url,
            "The PR state": pr.state,
            "The commit SHAs": get_sha(repo_),
        }
        #return pr_details_

tool_pr_details = FunctionTool.from_defaults(pr_details_tools,)

def files_tools(file_path_: str):
    """use this tool as following: given a file path as parameter,
        this tool can fetch the contents of the file
        from the repository."""
    repo_name_ = repo_url.split('/')[-1].replace('.git', '')
    username_ = repo_url.split('/')[-2]
    full_repo_name_ = f"{username_}/{repo_name_}"
    if git_ is not None:
        repo = git_.get_repo(full_repo_name_)
        file_content = repo.get_contents(file_path_).decoded_content.decode('utf-8')
        return file_content

tool_get_file = FunctionTool.from_defaults(files_tools,)


def pr_commit_details_tools(commit_sha: str):
    """given the commit SHA, this function can retrieve information
     about the commit, such as the files that changed,
     and return that information"""
    #git_= Github(os.getenv("GITHUB_TOKEN")) if os.getenv("GITHUB_TOKEN") else None
    repo_name_ = repo_url.split('/')[-1].replace('.git', '')
    username_ = repo_url.split('/')[-2]
    full_repo_name_ = f"{username_}/{repo_name_}"
    if git_ is not None:
        repo = git_.get_repo(full_repo_name_)
        commit = repo.get_commit(commit_sha)
        changed_files_: list[dict[str, any]] = []
        for f in commit.files:
            changed_files_.append({
                "filename": f.filename,
                "status": f.status,
                "additions": f.additions,
                "deletions": f.deletions,
                "changes": f.changes,
                "patch": f.patch,
            })
        return changed_files_

tool_pr_commit_details = FunctionTool.from_defaults(pr_commit_details_tools,)

async def add_context_to_state(ctx: Context, context_: str) -> str:
    """Useful for adding the context to the state."""
    current_state = await ctx.get("state")
    current_state["gathered_contexts"] = context_
    await ctx.set("state", current_state)
    return "State updated with report contexts. "

tool_add_context_to_state = FunctionTool.from_defaults(add_context_to_state,)

async def add_comment_to_state(ctx: Context, draft_comment_: str) -> str:
    """Useful for adding the comments to the state."""
    current_state = await ctx.get("state")
    current_state["draft_comment"] = draft_comment_
    await ctx.set("state", current_state)
    return "State updated with comments. "

tool_add_comment_to_state = FunctionTool.from_defaults(add_comment_to_state,)

async def add_final_review_to_state(ctx: Context, final_review: str) -> str:
    """Useful for adding the final review to the state."""
    current_state = await ctx.get("state")
    current_state["final_review"] = final_review
    await ctx.set("state", current_state)
    return "State updated with final reviews. "

tool_add_final_review_to_state = FunctionTool.from_defaults(add_final_review_to_state,)

async def post_comment_to_github(ctx: Context,pr_number:int=pr_number) -> str:
    """Useful for posting a comment to Git"""
    current_state = await ctx.get("state")
    final_review = current_state.get("final_review")
    repo_name_ = repo_url.split('/')[-1].replace('.git', '')
    username_ = repo_url.split('/')[-2]
    full_repo_name_ = f"{username_}/{repo_name_}"
    if git_ is not None:
        repo_ = git_.get_repo(full_repo_name_)
        pr = repo_.get_pull(int(pr_number))
        pr.create_review(body=final_review)
        return f"Posted review to PR #{pr_number}."
    return "GitHub or final review not available."
tool_post_comment_to_github = FunctionTool.from_defaults(post_comment_to_github,)



commentor_agent = FunctionAgent(llm=llm_commenter,
                                name="CommentorAgent",
                                description="Uses the context gathered by the context agent to draft a pull review comment comment.",
                                tools=[tool_add_comment_to_state],
                                system_prompt=commenter_agent_system_prompt,
                                can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"])

context_agent = FunctionAgent(llm=llm_context,
                             name="ContextAgent",
                             description="Gathers all the needed context ... ",
                              tools=[tool_get_file, tool_pr_details, tool_pr_commit_details,
                                     tool_add_context_to_state],
                             system_prompt=context_agent_prompt,
                             can_handoff_to=["CommentorAgent"]
                             )

review_and_posting_agent = FunctionAgent(llm=llm,
                             name="ReviewAndPostingAgent",
                             description="Review and post comment to Github ",
                              tools=[tool_add_final_review_to_state, tool_post_comment_to_github,],
                             system_prompt=review_agent_system_prompt,
                             can_handoff_to=["CommentorAgent"]
                             )

workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent,review_and_posting_agent],
    root_agent=commentor_agent.name,
    initial_state={
        "gathered_contexts": "",
        "draft_comment": "",
        "final_review": "",
    },
)


async def main():
    #query = input().strip()
    query = "Write a review for PR: " + pr_number
    prompt = RichPromptTemplate(query)

    handler = workflow_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\\n\\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")

if __name__ == "__main__":
    asyncio.run(main())
    git_.close()