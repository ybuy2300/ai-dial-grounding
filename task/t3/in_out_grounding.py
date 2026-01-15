import asyncio
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr, BaseModel, Field
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

# HOBBIES SEARCHING WIZARD
# Before implementation open the `flow.png` to see the flow of app.
# Searches users by hobbies and provides their full info in JSON format:
#   Input: I need people who love to go to mountains
#   Output:
#     rock climbing: [{full user info JSON},...],
#     hiking: [{full user info JSON},...],
#     camping: [{full user info JSON},...]
# ---
# 1. Since we are searching hobbies that persist in `about_me` section - we need to embed only user `id` and `about_me`!
#    It will allow us to reduce context window significantly.
# 2. Pay attention that every 5 minutes in system will be added new users and some will be deleted. We will at the
#    'cold start' add all users for current moment to vectorstor and with each user request we will update vectorstor,
#    we will remove deleted users and add new - it will also resolve the issue with consistency within this 2 services
#    and will reduce costs (we don't need on each user request load vectorstor from scratch and pay for it).
# 3. We ask LLM make NEE (Named Entity Extraction) https://cloud.google.com/discover/what-is-entity-extraction?hl=en
#    and provide response in format:
#    {
#       "{hobby}": [{user_id}, 2, 4, 100...]
#    }
#    It allows us to save significant money on generation, reduce time on generation and eliminate possible
#    hallucinations (corrupted personal info or removed some parts of PII (Personal Identifiable Information)). After
#    generation we also need to make output grounding (fetch full info about user and in the same time check that all
#    presented IDs are correct).
# 4. In response we expect JSON with grouped users by their hobbies.
# ---
# This sample is based on the real solution where one Service provides our Wizard with user request, we fetch all
# required data and then returned back to 1st Service response in JSON format.

SYSTEM_PROMPT = """You are a RAG-powered assistant that groups users by their hobbies.

## Flow:
Step 1: User will ask to search users by their hobbies etc.
Step 2: Will be performed search in the Vector store to find most relevant users.
Step 3: You will be provided with CONTEXT (most relevant users, there will be user ID and information about user), and 
        with USER QUESTION.
Step 4: You group by hobby users that have such hobby and return response according to Response Format

## Response Format:
{format_instructions}
"""

USER_PROMPT = """## CONTEXT:
{context}

## USER QUESTION: 
{query}"""


llm_client = AzureChatOpenAI(
    temperature=0.0,
    azure_deployment='gpt-4o',
    azure_endpoint=DIAL_URL,
    api_key=SecretStr(API_KEY),
    api_version=""
)


class GroupingResult(BaseModel):
    hobby: str = Field(description="Hobby. Example: football, painting, horsing, photography, bird watching...")
    user_ids: list[int] = Field(description="List of user IDs that have hobby requested by user.")


class GroupingResults(BaseModel):
    grouping_results: list[GroupingResult] = Field(description="List matching search results.")


def format_user_document(user: dict[str, Any]) -> str:
    # Return user id and about_me info.
    # Sample:
    # User:
    #   id: {id}
    #   About user: {about_me}
    # ---
    return f"User:\n  id: {user.get('id')}\n  About user: {user.get('about_me')}"



class InputGrounder:
    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.user_client = UserClient()
        self.vectorstore = None

    async def __aenter__(self):
        await self.initialize_vectorstore()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def initialize_vectorstore(self, batch_size: int = 50):
        """Initialize vectorstore with all current users."""
        print("🔍 Loading all users for initial vectorstore...")

        # 1. Get all users (use UserClient)
        users = await self.user_client.get_all_users()

        # 2. Iterate through users and prepare array of Document with: `[Document(id=user.get('id'), page_content=format_user_document(user)) for user in users]`
        #    Pay attention that we save user id in separate column, we will use it later for removal of deleted users.
        #    Also, we persint in `page_content` just user id and `about_me` content, not the whole JSON.
        for user in users:
            documents = [Document(id=str(user.get('id')), page_content=format_user_document(user)) for user in users]
        
        # 3. Split all `documents` on batches (100 documents in 1 batch). We need it since Embedding models have limited context window
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        print(f"↗️ Creating embeddings and vectorstore for {len(documents)} documents...")
        
        # 4. Setup vectorstore:
        #       - create Chroma (FAISS doesn't support necessary functionality for further task) as `self.vectorstore` with
        #           - collection_name="users"
        #           - embedding_function=self.embeddings
        #       - Prepare tasks array: iterate through batches and call `self.vectorstore.aadd_documents(batch)`
        #       - Gather tasks: `await asyncio.gather(*tasks)`
        self.vectorstore = Chroma(
            collection_name="users",
            embedding_function=self.embeddings
        )
        tasks = []
        for batch in batches:
            tasks.append(self.vectorstore.aadd_documents(batch))
        await asyncio.gather(*tasks)
        print("✅ Vectorstore is ready.")


    async def retrieve_context(self, query: str, k: int = 100, score: float = 0.2) -> str:
        """Retrieve context, with optional automatic vectorstore update."""
        # 1. Call `_update_vectorstore` to fetch new users to vectorstor and remove deleted
        await self._update_vectorstore()

        # 2. Make similarity search (`similarity_search_with_relevance_scores` method)
        relevant_docs_with_scores = await self.vectorstore.similarity_search_with_relevance_scores(query, k=k)
        print("Retrieved context:")

        # 3. Create `context_parts` empty array (we will collect content here)
        context_parts = []
        
        # 4. Iterate through retrieved relevant docs (pay attention that its tuple (doc, relevance_score)) and:
        #       - add doc page content to `context_parts` and then `print(f"Retrieved (Score: {relevance_score:.3f}): {doc.page_content}")`
        for doc, relevance_score in relevant_docs_with_scores:
            context_parts.append(doc.page_content)
            print(f"Retrieved (Score: {relevance_score:.3f}): {doc.page_content}")
        
        # 5. Return joined context from `context_parts` with `\n\n` spliterator (to enhance readability)
        return "\n\n".join(context_parts)

    async def _update_vectorstore(self):
        # 1. Get all users (use UserClient)
        users = await self.user_client.get_all_users()
        
        # 2. Get all the data from the vectorstore: `self.vectorstore.get()`
        vectorstore_data = self.vectorstore.get()
        
        # 3. Get set of ids from the vectorstor: `set(str(user_id) for user_id in vectorstore_data.get("ids", []))`. We
        #    need it to compare ids from DB with ids that we get via latest API call to UserService
        set_vectorstore_ids = set(str(user_id) for user_id in vectorstore_data.get("ids", []))
        
        # 4. Prepare dict from retrieved users (key is user id, value is full user info): `{str(user.get('id')): user for user in users}`
        users_dict = {str(user.get('id')): user for user in users}
        # 5. Prepare set with users ids
        users_ids_set = set(users_dict.keys())
        # 6. Find new user ids: `users_ids_set - vectorstore_ids_set`
        new_user_ids = users_ids_set - set_vectorstore_ids
        # 7. Find user ids that need to be deleted: `vectorstore_ids_set - users_ids_set`
        ids_to_delete = list(set_vectorstore_ids - users_ids_set)
        # 8. If `ids_to_delete` is not empty then delete from vectorstore all rows with collected `ids_to_delete`.
        #    Chroma has method `delete`, that applies list of ids
        if ids_to_delete:
            await self.vectorstore.adelete(ids=ids_to_delete)
        # 9. Prepare new Documents:
        #       - Iterate through new user ids and create array with Documents: `Document(id=user_id, page_content=format_user_document(users_dict[user_id]))`
        new_documents = [Document(id=user_id, page_content=format_user_document(users_dict[int(user_id)])) for user_id in new_user_ids]

        # 10. Id new documents are present then save them to vectorstore
        if new_documents:
            await self.vectorstore.aadd_documents(new_documents)


    def augment_prompt(self, query: str, context: str) -> str:
        # Make augmentation for USER_PROMPT via `format` method
        return USER_PROMPT.format(context=context, query=query)

    def generate_answer(self, augmented_prompt: str) -> GroupingResults:
        # 1. Create PydanticOutputParser with `pydantic_object=GroupingResults` as `parser`
        parser = PydanticOutputParser(pydantic_object=GroupingResults)
        # 2. Create messages array with:
        #       - SystemMessagePromptTemplate.from_template(template=SYSTEM_PROMPT)
        #       - HumanMessage(content=augmented_prompt)
        messages = [
            SystemMessagePromptTemplate.from_template(template=SYSTEM_PROMPT).partial(format_instructions=parser.get_format_instructions()),
            HumanMessage(content=augmented_prompt)
        ]
        # 3. Generate `prompt`: `ChatPromptTemplate.from_messages(messages=messages).partial(format_instructions=parser.get_format_instructions())`
        prompt = ChatPromptTemplate.from_messages(messages=messages).partial(format_instructions=parser.get_format_instructions())
        # 4. Invoke it: `(prompt | llm_client | parser).invoke({})` as `grouping_results: GroupingResults`
        grouping_results: GroupingResults = (prompt | llm_client | parser).invoke({})
        # 5. return grouping_results
        return grouping_results


class OutputGrounder:
    def __init__(self):
        self.user_client = UserClient()

    async def ground_response(self, grouping_results: GroupingResults):
        # 1. Iterate through grouping results
        for grouping_result in grouping_results.grouping_results:
            # 2. Print hobby
            print(f"Hobby: {grouping_result.hobby}")
            # 3. Print fetched users: await self._find_users(grouping_result.user_ids)
            users = await self._find_users(grouping_result.user_ids)
            print(f"Users: {users}")

    async def _find_users(self, ids: list[int]) -> list[dict[str, Any]]:
        async def safe_get_user(user_id: int) -> Optional[dict[str, Any]]:
            try:
                # Get user by id (it is async method)
                user = await self.user_client.aget_user(user_id)
                return user
            except Exception as e:
                if "404" in str(e):
                    print(f"User with ID {user_id} is absent (404)")
                    return None
                raise  # Re-raise non-404 errors

        # 1. Prepare task array to get users and gather results
        tasks = [safe_get_user(user_id) for user_id in ids]
        results = await asyncio.gather(*tasks)

        # 2. Filter results and provide users that is not None        
        return [user for user in results if user is not None]


async def main():
    embeddings = AzureOpenAIEmbeddings(
        deployment='text-embedding-3-small-1',
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        dimensions=384,
        check_embedding_ctx_length=False
    )
    output_grounder = OutputGrounder()

    async with InputGrounder(embeddings, llm_client) as rag:
        print("Query samples:")
        print(" - I need people who love to go to mountains")
        print(" - Find people who love to watch stars and night sky")
        print(" - I need people to go to fishing together")

        while True:
            user_question = input("> ").strip()
            if user_question.lower() in ['quit', 'exit']:
                break

            # 1. Retrieve context
            context = await rag.retrieve_context(user_question)
            # 2. Make augmentation
            augmented_prompt = rag.augment_prompt(user_question, context)
            # 3. Generate answer
            grouping_results = rag.generate_answer(augmented_prompt)
            # 4. Make output grounding
            await output_grounder.ground_response(grouping_results)


if __name__ == "__main__":
    asyncio.run(main())
