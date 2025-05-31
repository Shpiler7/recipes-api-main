import asyncio
from typing import List, Dict, Any

from llama_index.core.agent.workflow import AgentOutput, ToolCall, ToolCallResult, FunctionAgent, AgentWorkflow
from llama_index.core.workflow import Context
from llama_index.core.prompts import RichPromptTemplate
from github import Github
from llama_index.llms.openai import OpenAI
import dotenv
import os

dotenv.load_dotenv(dotenv_path="C:\\Test\\.env")

llm = OpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
)

git = Github(os.getenv("GITHUB_TOKEN")) if os.getenv("GITHUB_TOKEN") else None

repo_url = "https://github.com/Shpiler7/recipes-api-main.git"
repo_name = repo_url.split('/')[-1].replace('.git', '')
username = repo_url.split('/')[-2]
full_repo_name = f"{username}/{repo_name}"
repo = git.get_repo(full_repo_name)


# if git is not None:
#
#     file_content =


async def get_pr_details(ctx: Context, pr_number: int) -> dict[str, Any]:
    """Useful for getting PR details such as author, title, body, diff URL, state and commit SHAs."""
    pull_request = repo.get_pull(pr_number)

    commit_SHAs = []
    commits = pull_request.get_commits()

    for c in commits:
        commit_SHAs.append(c.sha)

    details: dict[str, any] = {
        "author": pull_request.user.login,
        "title": pull_request.title,
        "body": pull_request.body,
        "diffURL": pull_request.diff_url,
        "state": pull_request.state,
        "commitSHAs": commit_SHAs
    }

    return details


async def get_pr_commit_details(ctx: Context, sha: str) -> list[dict[str, Any]]:
    """Useful for getting the commit details based on SHA, such as filename, status, additions, deletions,
    changes and patch (diff)"""

    commit = repo.get_commit(sha)
    changed_files: list[dict[str, any]] = []
    for f in commit.files:
        changed_files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
            "patch": f.patch,
        })

    return changed_files


async def get_file_details(ctx: Context, file_path: str) -> str:
    """Useful for getting details of the specific file in pull request."""
    return repo.get_contents(file_path).decoded_content.decode('utf-8')


async def save_pr_details(ctx: Context, pr_details: str):
    """Useful for saving gathered contexts."""
    current_state = await ctx.get("state")
    current_state["gathered_contexts"] = pr_details
    await ctx.set("state", current_state)


async def save_draft_comment(ctx: Context, draft_comment: str):
    """Useful for saving draft comment."""
    current_state = await ctx.get("state")
    current_state["draft_comment"] = draft_comment
    await ctx.set("state", current_state)


async def save_final_review(ctx: Context, final_review: str):
    """Useful for saving final comment."""
    current_state = await ctx.get("state")
    current_state["final_review_comment"] = final_review
    await ctx.set("state", current_state)


async def post_final_review(ctx: Context, final_review: str, pr_number: int):
    """Useful for posting review to GitHub."""
    pull_request = repo.get_pull(pr_number)
    pull_request.create_review(body=final_review)


context_agent = FunctionAgent(
    name="ContextAgent",
    description="Useful for retrieving Context of the PR.",
    system_prompt=("""You are the context gathering agent. When gathering context, you MUST gather \n: 
                    - The details: author, title, body, diff_url, state, and head_sha; \n
                    - Changed files; \n
                    - Any requested for files; \n
                    Once you gather the requested info, you MUST hand control back to the Commenter Agent."""),
    llm=llm,
    tools=[get_pr_details, get_pr_commit_details, get_file_details, save_pr_details],
    can_handoff_to=["CommentorAgent"]
)

commenter_agent = FunctionAgent(
    name="CommentorAgent",
    description="Useful for commenting creating a draft comment.",
    system_prompt=("""You are the commenter agent that writes review comments for pull requests as a human reviewer 
    would. \n Ensure to do the following for a thorough review: - Request for the PR details, changed files, 
    and any other repo files you may need from the ContextAgent. - Once you have asked for all the needed 
    information, write a good ~200-300 word review in markdown format detailing: \n - What is good about the PR? \n - 
    Did the author follow ALL contribution rules? What is missing? \n - Are there tests for new functionality? If 
    there are new models, are there migrations for them? - use the diff to determine this. \n - Are new endpoints 
    documented? - use the diff to determine this. \n - Which lines could be improved upon? Quote these lines and 
    offer suggestions the author could implement. \n - If you need any additional details, you must hand off to the 
    Commenter Agent. \n - You should directly address the author. So your comments should sound like: \n ''Thanks for 
    fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?
    - You must create draft review.
    - You must hand off to the ReviewAndPostingAgent once you are done drafting a review.'"""),
    llm=llm,
    tools=[save_draft_comment],
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"]
)

review_and_post_agent = FunctionAgent(
    name="ReviewAndPostingAgent",
    description="Useful for reviewing the comment and posting review.",
    system_prompt=("""You are the Review and Posting agent. You must use the CommentorAgent to create a review 
    comment. Once a review is generated, you need to run a final check and post it to GitHub. - The review must: \n - 
    Be a ~200-300 word review in markdown format. \n - Specify what is good about the PR: \n - Did the author follow 
    ALL contribution rules? What is missing? \n - Are there notes on test availability for new functionality? If 
    there are new models, are there migrations for them? \n - Are there notes on whether new endpoints were 
    documented? \n - Are there suggestions on which lines could be improved upon? Are these lines quoted? \n If the 
    review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. \n 
    When you are satisfied, post the review to GitHub. '"""),
    llm=llm,
    tools=[save_final_review, post_final_review],
    can_handoff_to=["CommentorAgent"]
)

workflow_agent = AgentWorkflow(
    agents=[context_agent, commenter_agent, review_and_post_agent],
    root_agent=review_and_post_agent.name,
    initial_state={
        "gathered_contexts": "",
        "draft_comment": "",
        "final_review_comment": ""
    },
)


async def main():
    query = input().strip()
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
    git.close()
