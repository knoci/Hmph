"""Isolated fixed-author TiMem components; never imported by the product runtime.

Only provider dispatch, transport and local storage/embedding configuration are
adapted. Author generation/retrieval methods and prompts remain unchanged.
"""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

TIMEM_REVISION = "7e05ffda855602a378d4f4a45a3e97e1d1ce4cb9"


def local_test_key(root):
    """Read paid experiment credentials only from the ignored local file."""
    from dotenv import dotenv_values
    from pydantic import SecretStr

    value = dotenv_values(Path(root) / ".env.local").get("HMPH_DEEPSEEK_API_KEY")
    if not value:
        raise ValueError("Ignored local DeepSeek test key unavailable")
    return SecretStr(value)


class MeteredTiMemLLM:
    def __init__(self, key, budget, case_id, *, max_tokens=1024, stop_file=None):
        import httpx
        from openai import AsyncOpenAI

        from hmph.reference_transport import AsyncReferenceTransport

        self.transport = AsyncReferenceTransport(
            key,
            max_calls=budget.limits["per_case_calls"],
            max_tokens=max_tokens,
            budget=budget.bind(case_id, "timem", "writer"),
            stop_file=stop_file,
        )
        self.http = httpx.AsyncClient(
            transport=self.transport, timeout=60, trust_env=False, follow_redirects=False
        )
        self.sdk = AsyncOpenAI(
            api_key="native-placeholder",
            base_url="https://api.deepseek.com",
            http_client=self.http,
            max_retries=0,
        )
        self.failed = False
        self.failure_reason = None
        self.max_tokens = max_tokens

    async def chat(self, messages, **kwargs):
        from hmph.reference_transport import ReferenceCallError

        if self.failed:
            raise ReferenceCallError("reference_cancelled")
        try:
            result = await self.sdk.chat.completions.create(
                model="deepseek-flash",
                temperature=0,
                max_tokens=self.max_tokens,
                messages=[{"role": m.role.value, "content": m.content} for m in messages],
            )
        except BaseException as error:
            self.failed = True
            from hmph.reference_transport import failure_code

            cause = error
            for _ in range(4):
                if isinstance(cause, ReferenceCallError) or cause.__cause__ is None:
                    break
                cause = cause.__cause__
            self.failure_reason = failure_code(cause)
            raise
        from llm.base_llm import ChatResponse

        return ChatResponse(
            content=result.choices[0].message.content,
            finish_reason="stop",
            model=result.model,
            usage=result.usage.model_dump(),
            response_time=0,
        )

    async def chat_stream(self, messages, **kwargs):
        # Native generator expects an async iterator; the metered provider buffers
        # one nonstreaming response. No unmetered SDK or paid retry is involved.
        yield (await self.chat(messages, **kwargs)).content

    async def complete(self, prompt, **kwargs):
        from llm.base_llm import Message, MessageRole

        return (await self.chat([Message(MessageRole.USER, prompt)])).content

    async def close(self):
        await self.http.aclose()


def install_provider(llm):
    """Register one external provider without importing unused GPU adapters."""
    # Upstream llm/__init__.py eagerly imports the GPU embedding implementation.
    # Load its namespace without those optional exports; base_llm remains native.
    specification = importlib.util.find_spec("llm")
    if specification is None:
        raise ImportError("Fixed TiMem installation missing")
    package = types.ModuleType("llm")
    package.__path__ = list(specification.submodule_search_locations)
    package.__spec__ = specification
    sys.modules["llm"] = package
    dispatch = types.ModuleType("llm.llm_manager")
    dispatch.get_llm = lambda provider=None: llm
    package.get_llm = dispatch.get_llm
    sys.modules[dispatch.__name__] = dispatch


def leaf_sources(node_id, nodes, leaf_map):
    """Resolve author child links strictly; an unmapped child is a coverage gap."""
    visiting, result = set(), set()

    def walk(identity):
        if identity in visiting:
            raise ValueError("Cyclic native provenance")
        if identity in leaf_map:
            result.update(leaf_map[identity])
            return
        node = nodes.get(identity)
        if node is None or not node.child_memory_ids:
            raise ValueError("Missing native provenance")
        visiting.add(identity)
        for child in node.child_memory_ids:
            walk(child)
        visiting.remove(identity)

    walk(node_id)
    return tuple(sorted(result))


class LocalTiMem:
    """Author generator/vector adapter/planner with a local Qdrant collection.

    This is a component experiment, not the author's complete scheduler or a
    product backend. PG/Redis and production cleanup are intentionally absent.
    """

    async def open(self, directory: Path):
        from fastembed import TextEmbedding
        from qdrant_client import QdrantClient, models

        model = TextEmbedding(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            cache_dir="artifacts/models/multilingual-minilm",
            local_files_only=True,
        )

        async def embed(text):
            return (await asyncio.to_thread(lambda: list(model.embed([text]))))[0].tolist()

        embedding = types.ModuleType("llm.embedding_service")
        embedding.get_embedding_service = lambda: types.SimpleNamespace(embed_text=embed)
        sys.modules[embedding.__name__] = embedding
        from storage.vector_adapter import VectorAdapter
        from storage.vector_store import VectorStore
        from timem.memory.memory_generator import MemoryGenerator

        self.generator = MemoryGenerator()
        self.generator.max_retries = 1

        async def once(operation, layer, name):
            return await asyncio.wait_for(operation(), timeout=60)

        # Both upstream helpers contain unconditional while True and ignore the
        # configuration above. Replace retry orchestration, never generation.
        self.generator._retry_with_exponential_backoff = once
        self.generator._retry_with_exponential_backoff_for_streaming = once
        # Author prompt persistence can contain full inputs; research needs only
        # original public fixture + code/prompt hashes, never extra prompt copies.
        self.generator._append_prompt_log = lambda *args, **kwargs: None
        self.nodes, self.leaf_map = {}, {}
        self.client = QdrantClient(path=str(directory / "qdrant"))
        self.collection = "hmph_timem_research"
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        )
        config = {"collection_name": self.collection, "vector_size": 384, "distance": "Cosine"}
        # Native adapter methods are unchanged; construct with its public config
        # seam, then attach the official Qdrant embedded client to its store.
        manager = types.SimpleNamespace(reload_config=lambda: None, get_config=lambda key: config)
        self.adapter = VectorAdapter(config_manager=manager)
        self.adapter.vector_store = VectorStore(config)
        self.adapter.vector_store.client = self.client
        self.adapter.vector_store._initialized = True
        self.adapter._is_available = True

    async def add(self, content, *, level, when, children=(), sources=(), session_id=None):
        from timem.models.memory import Memory

        start = min([when] + [self.nodes[c].time_window_start for c in children])
        end = max([when] + [self.nodes[c].time_window_end for c in children])
        node = Memory(
            user_id="research",
            expert_id="hmph",
            level=level,
            title=f"{level} research",
            content=content,
            time_window_start=start,
            time_window_end=end,
            created_at=when,
            updated_at=when,
            child_memory_ids=list(children),
            session_id=session_id,
        )
        await self.adapter.store_memory(node, wait_for_vector_indexing=True)
        self.nodes[node.id] = node
        if sources:
            self.leaf_map[node.id] = tuple(sources)
        return node

    async def retrieve(self, question):
        from timem.workflows.retrieval_nodes.hierarchical_retriever import HierarchicalRetriever
        from timem.workflows.retrieval_nodes.retrieval_planner import RetrievalPlanner
        from timem.workflows.retrieval_state import RetrievalStrategy

        state = await RetrievalPlanner().run({"question": question})
        if state.get("errors"):
            raise ValueError("Native planner failed")
        # Explicit component profile: native rule planning + L1/L2 hierarchical
        # retrieval. Not the full author multi-strategy or LLM planner pipeline.
        state["selected_strategies"] = [RetrievalStrategy.HIERARCHICAL]
        storage = types.SimpleNamespace(
            search_memories=lambda query, options, kind: self.adapter.search_memories(
                query, options
            )
        )
        state = await HierarchicalRetriever(storage_manager=storage).run(state)
        if state.get("errors"):
            raise ValueError("Native hierarchical retrieval failed")
        rows = state.get("hierarchical_results", [])
        ids = [r.get("id") or r.get("memory_id") for r in rows]
        return ids, sorted({s for n in ids for s in leaf_sources(n, self.nodes, self.leaf_map)})

    def close(self):
        self.client.delete_collection(self.collection)
        empty = self.client.get_collections().collections == []
        self.client.close()
        return empty
