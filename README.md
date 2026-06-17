<p align="right">
  <strong>🇬🇧 English</strong> |
  <a href="README.de.md">🇩🇪 Deutsch</a>
</p>

# VIOLET — **V**irtual **I**ntelligence **O**bserving to **L**earn, **E**volving to **T**hink

**Meet Violet** - a locally hosted AI "companion" with persistent memory, dynamic emotional state, and a modular architecture designed to grow over time. The project separates core functions radically: the language model itself is just the voice. Everything else is built around it, mirroring the human brain.

---

## Architecture Overview

AMNIVUM is the technical stack that "produces" Violet. It consists of independent modules communicating asynchronously via a central message queue. No module calls another directly — they publish and subscribe, which keeps latency predictable and the system extensible.

```
                       ┌───────────────┐
                       │     SENSUS    │  Perception layer (audio, video, STT)
                       └───────┬───────┘
                               │
                       ┌───────▼───────┐
                       │     KORTEX    │  Prompt assembler — LLM-facing module
                       └───┬──────┬────┘
                           │      │
              ┌────────────▼─┐  ┌─▼────────────┐
              │   MEMORIA    │  │   INGENIUM   │  Memory and Emotions run in parallel per turn
              └────────────┬─┘  └─┬────────────┘
                           │      │
                       ┌───▼──────▼───┐
                       │     LLM      │  Language center
                       └──────────────┘
```

Each module is a Python asyncio task. The shared message queue is the only communication channel.

*The project ist still under heavy Development. Architecture is mostly planned and specified. Implementation has started for some Modules. Changes can and will happen as the project takes shape.*

**Current Code: 984 lines Python**

---

## Modules

### KORTEX — Prompt Assembler

KORTEX is the orchestrator of the turn cycle. It receives plain text from the user over SENSUS, distributes it to MEMORIA and INGENIUM, collects their outputs, and assembles the final prompt for the language model. On the output side it unpacks the LLM response and routes it symmetrically back through the same pipeline.

KORTEX does not interpret — it assembles. All semantic and emotional context is provided by other modules.

**Status:** Architecture 99% specified. Implementation at 25% (Input handler experimental).

---

### MEMORIA — Memory System

MEMORIA manages everything Violet knows and remembers. It operates on three layers with distinct lifetimes, and runs a chunking and retrieval pipeline on every turn.

#### Storage Layers

**MEMORIA-LONG** — persistent vector storage (Qdrant Collection 1)  
Consolidated long-term knowledge about the user, Violet herself, and the world. Written exclusively during offline consolidation. Entries marked `core: true` are decay-immune. All others follow a configurable decay function. Retrieval is thematic via cosine similarity over three main topics.

**MEMORIA-MID** — session-spanning temporary storage (Qdrant Collection 2)  
Chunks from the current and previous sessions, including emotion tags assigned by INGENIUM. No decay — cleared completely during consolidation. Provides recent conversational context and contradiction detection material.

**MEMORIA-SHORT** — in-memory context buffer (RAM only)  
The active context window for KORTEX. Holds approximately 20 chunks, dynamically selected and re-evaluated after every turn. Never persisted. Structured as a ranked relevance queue from LONG and MID, with guaranteed representation across knowledge types (user knowledge, world knowledge, AI self-knowledge).

#### Buffer Slot Logic

The buffer fills from a two-pass template. Pass 1 guarantees minimum representation per knowledge source (user, world, AI, MID), redistributing unused capacity downstream. Pass 2 fills remaining slots dynamically via round-robin across available pools. Topics 2 and 3 (satellite clusters) receive a fixed side allocation. The result is a buffer that is always as full as available data allows, without artificially inflating any single source.

#### Chunking Pipeline

Input text is split at sentence boundaries via regex, embedded with a multilingual sentence transformer, then semantically merged: adjacent sentences above a cosine similarity threshold are joined as long as the combined word count stays within bounds. Short residual chunks are attached to neighbors rather than discarded. The merged embedding is only an approximation — the final Qdrant embedding is computed fresh from the merged text.

#### Retrieval

Per turn, the input chunks are clustered by topic using agglomerative clustering on their embeddings. One weighted average vector is computed per topic (up to three). Each topic vector is matched against the cluster graph to find the corresponding LONG cluster. Retrieval then runs three parallel queries per cluster — one per knowledge source — to guarantee representation, followed by a broader similarity sweep for remaining candidates. MID is queried separately by similarity and recency.

The cluster graph is a JSON index of all LONG cluster centroids and their neighbor relationships, maintained offline by the consolidator. It enables satellite cluster selection at runtime without additional Qdrant queries.

#### Importance Gate

Before a turn chunk is written to MID, three conditions are checked simultaneously: the chunk is too short, it is too similar to existing cluster centroids (redundant), and its emotion signal is flat (high neutral score, low amplitude). If all three apply, the chunk is not written to MID. It always lands in the session log — the offline consolidator sees everything.

#### Topic Switch Detection

After every turn, the turn embedding is compared against a sliding window average of recent turn embeddings. If similarity drops below a threshold, a topic switch is detected and the buffer is reloaded with fresh Qdrant queries against the new topic vector.

#### Offline Processes

**Consolidator** — runs during LLM idle time, analogous to humans REM sleep. Reads the session log, chunks and embeds it, passes it to INGENIUM-OFFLINE for importance scoring and contradiction detection, then selectively writes to MEMORIA-LONG. Clears MEMORIA-MID completely on completion.

**Eraser** — runs periodically. Applies the decay function to LONG entries and removes those below the importance threshold. Core entries are untouched.

**Status:** Architecture 99% specified. Implementation at 50% (Online pipeline - chunking, retrieval, buffer, importance gate, topic switch) experimental. Offline processes (consolidator, eraser) specified.

---

### INGENIUM — Personality and Affect

INGENIUM gives Violet a persistent emotional state that influences every response without being hardcoded into any prompt.

**INGENIUM-STATIC** — a character LoRA merged into the base language model. Defines traits, temperament, response tendencies and prompt format. Not modifiable at runtime — requires retraining.

**INGENIUM-INTERPRETER** — a multilingual emotion classifier (XLM-RoBERTa-based, 11 emotion dimensions). Classifies every input chunk and every LLM output chunk symmetrically, producing a probability distribution across all 11 labels per chunk.

**INGENIUM-AFFECT** — a persistent JSON state file containig `global_affect`, a slow-moving emotional baseline accumulated across all turns

The affect state is updated in two passes per turn: once before the prompt is assembled (using retrieved clean tags), and once after KORTEX has the full picture (using turn tags, raw tags, and acceptance tags weighted together).

Drift detection compares incoming emotion vectors against cluster history. Low drift validates the existing affect. High drift raises a conflict flag to KORTEX, which can trigger a clarification sequence from the LLM.

**Status:** Architecture 99% specified. Implementation at 25% (Interpreter experimental).

---

### SENSUS — Perception Layer

SENSUS handles all peripheral input and output. It is structured in three tiers: CLIENT (microphone, camera, speaker), NODE (STT via Whisper, avatar rendering, voice synthesis), and HOST (metadata interpretation, KORTEX interface). Audio and video streams flow inward as structured text descriptions, and KORTEX output flows outward as rendered audio and avatar motion. When finished, it will provide different interaction modes such as chat-only, voice, and video+voice, depending on hardware availability and system capabilities.

**Status:** Architecture planned. Implementation not started.

---

### ANIMUS — Inner Voice

ANIMUS is Violet's autonomous inner life — a second LLM process that runs outside of user turns, on its own timing. It has full access to MEMORIA and INGENIUM, and communicates back into the main stack the same way KORTEX does.

The active half of ANIMUS produces output: unprompted messages, questions, reactions to ambient SENSUS input, and signals that shift VIOLET's affect state without any user action. The passive half, developed over much longer timescales, is intended to gradually replace deterministic decisions in MEMORIA and INGENIUM with trained intuition — what to remember and how to feel about something — built from the accumulated interaction data NUCLEUS produces naturally over thousands of turns.

**Status:** Architecture still in planning. Implementation starts after core stack is stable.

---

## Language Model

The LLM is not Violet's brain — it is her voice. Context is carried entirely through MEMORIA's buffer. Emotional framing is provided entirely by INGENIUM. KORTEX assembles a structured prompt that includes affect tags, chunk-level emotion metadata, and flags for conflicts or knowledge gaps. The model is expected to speak from what it is given, not to infer from what it knows.

A LoRA adapter trains the model to treat this prompt format as native, to ask rather than assume when context is missing, and to maintain a consistent character voice. The base model is selected for quality (german) language output, instruction-following capability, and receptiveness to fine-tuning — in the 3–7B parameter range, uncensored.

**Status:** Model selection and testing in progress. LoRA training data strategy defined. Implementation not started.

---

## Tech Stack

| Component | Technology | Hardware |
|---|---|---|
| Runtime | Python asyncio | CPU |
| Vector Storage | Qdrant (2 collections) | CPU / RAM |
| Embedding | paraphrase-multilingual-MiniLM-L12-v2 | CPU |
| Emotion Classifier | multilingual-emotion-classification (XLM-RoBERTa) | CPU |
| LLM Backend | llama.cpp or Ollama | GPU |
| Infrastructure | Docker Compose | — |

---

## Project Structure

```
AMNIVUM/
├── nucleus/                  # Main Python package
│   ├── kortex/               # Prompt assembly, input/output handling
│   ├── memoria/              # Memory system (core, retriever, short)
│   ├── ingenium/             # Affect state, emotion classifier
│   ├── sensus/               # Perception layer
│   ├── animus/               # Inner voice
│   └── shared/               # Messages, queues, services, config
├── data/                     # Runtime persistent data (not versioned)
│   ├── affect.json
│   ├── cluster_graph.json
│   └── session_log/
├── vector_storage/           # Qdrant data directory (not versioned)
├── docker-compose.yml        # Infrastructure definition
├── .env                      # All configuration and secrets (not versioned)
└── pyproject.toml
```

---

## Next Open Points

- [ ] Finalize Qdrant collection schema and metadata fields
- [ ] Calibrate all similarity thresholds empirically against real data
- [ ] Implement INGENIUM affect update into the turn cycle
- [ ] Offline consolidator and eraser