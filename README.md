# Lit Lake: Search & Analyze Literature Libraries with Claude

> ⚠️ **UNDER CONSTRUCTION** ⚠️  
> This project (and the README itself for that matter) is actively being developed. Features may be incomplete, documentation may be outdated, and breaking changes may occur. Use at your own risk and check back for updates!

> Runtime update: Lit Lake now includes a Python + UV MCPB server implementation with durable queue/audit tables (`jobs`, `job_attempts`, `worker_runs`).

Connect [Claude](https://www.anthropic.com/claude) (and others) to [Zotero](https://www.zotero.org/) with one click. Discover related papers through semantic search, analyze references with full context and allow Claude to access the full text (including embedded images) of your library.

## Installation
### Claude (recommended)
Claude is the easiest and fastest way to get set up with Lit Lake. If you do not have Claude installed on your computer, download it [here](https://claude.com/download).

> If you're interested in learning more about what an `.mcpb` file is, read Anthropic's blog post [here](https://www.anthropic.com/engineering/desktop-extensions)!

#### Via MCPB
1. If you do not have Claude desktop installed, install it [here](https://claude.com/download)
2. Click [here](https://github.com/ElliotRoe/lit-lake/releases/latest) to view the latest version of Lit Lake
3. Click the file that ends with `.mcpb`
4. Wait for it to download
5. Double click the downloaded file
6. Click 'Install'

> Note: If double clicking does not immediately bring you to the installation screen within Claude, go to Settings > Extensions > Advanced settings > Install Extension

#### Configuration
During installation, you can configure the following options. All are **optional** with sensible defaults.

| Option | Description | Default |
|--------|-------------|---------|
| **Zotero DB** | Path to your `zotero.sqlite` file | `~/Zotero/zotero.sqlite` |
| **Lit Lake Folder** | Parent folder where the `LitLake` data directory will be created | `~/LitLake` |
| **Extraction Backend** | Chooses extraction backend (`local` or `gemini`) | `local` |
| **Gemini API Key** | Enables higher-quality PDF extraction via Gemini | Local extraction (no API needed) |
| **Disable Embedding** | Turns off background embedding processes to save battery | Off (embedding runs) |

**Zotero DB**: Only set this if you've moved your Zotero data directory to a custom location. Examples:
- **macOS/Linux**: `/Users/yourname/Custom/Zotero/zotero.sqlite`
- **Windows**: `C:\Users\yourname\Zotero\zotero.sqlite`

**Lit Lake Folder**: If you select a folder that is not named `LitLake`, the app will create a `LitLake` subdirectory inside it. This is where your database and AI models are stored.

**Extraction Backend**: Set `EXTRACTION_BACKEND=local` (default) or `EXTRACTION_BACKEND=gemini`. Lit Lake uses one backend per run and does not mix backends.

**Gemini API Key**: Required only when `EXTRACTION_BACKEND=gemini`. If set, Lit Lake uses Gemini for PDF extraction.

**Disable Embedding**: Enable this option to stop background embedding processes. Useful when you're on battery power and want to conserve energy. Note: semantic search won't work for new documents until embeddings are generated.

#### First Run
On first launch, Lit Lake will:
1. **Automatically sync your Zotero library** — no manual action needed
2. **Download AI models** (~500MB total) — this happens once and may take a few minutes
3. **Begin extracting full text from supported attachments** — runs in the background, progress visible via `library_status`

You can check progress by asking Claude to call `library_status`. Once `init_status` shows "Ready", semantic search is available.

> **Note**: If you enabled "Disable Embedding" for battery savings, semantic search won't work until you disable that option and let the embedding process run.

#### Attachment Full-Text Extraction
Lit Lake automatically extracts text from supported attachment types (including PDF and HTML web snapshots) so Claude can search and analyze full papers and saved webpages — not just titles and abstracts.

**How it works:**
- **By default**, Lit Lake uses local extraction (`pypdf` for PDFs + `trafilatura` for HTML snapshots). No API keys required.
- **With `EXTRACTION_BACKEND=gemini`**, extraction uses Google's Gemini model for PDFs.

**What happens during extraction:**
1. Text is extracted from each supported attachment
2. Raw text is normalized (line wraps and hyphenation fixed) to improve quality
3. Full text is stored in the database for direct access
4. Text is split into searchable chunks with embeddings for semantic search
5. When available, location metadata (like PDF page ranges) is stored in `documents.metadata_json` for citation-friendly references

Check extraction progress via `library_status` — look for the `extraction` status counts.
Location metadata is optional and backend-dependent. Backends that do not return it leave `metadata_json.loc` empty.

### Other LLMs
To use Lit Lake with other LLM clients (like LM Studio, Cherry studio, etc), you'll just need to download the binary file and make it executable then configure it globally. Honestly, I haven't configured it yet with another client, if you are attempting, please reach out and I can help, then I'll add the instructions back here.

## Queue Maintenance

- `uv run lit-lake-queue stats`  
  Shows durable queue depth by queue/state.
## Use Cases

#### Paper Discovery & Search 
The search ability is great for discovering individual or groups of papers based on a topic. Claude can search for keywords and search semantically within the titles, abstracts, and full-text content of your references. Due to how this tool is implemented, Claude can combine these search types in arbitrary and programmatic ways.

*Examples*

**Keyword Search** 
> "Can you find all articles in my library that have the keyword 'pedagogical content knowledge' or 'PCK' within their abstract?"

**Semantic Search** 
> "Can you find me articles in my library related to out of field teaching?"

**Advanced Search** 
> "Find me all articles that have keywords 'pedagogical content knowledge' or 'PCK' within their abstract and are related to out of field teaching or teaching self-efficacy."

#### Full-Text Analysis
With attachment extraction enabled, Claude can access and analyze the complete content of your papers and snapshots — not just titles and abstracts. This is invaluable for verification, deep research, and finding specific passages.

*Examples*

**Find Specific Passages**
> "What does Smith 2023 say about their methodology for data collection?"

**Search Full Text Across Library**
> "Find passages in my library that discuss the limitations of self-report measures."

**Verify Claims**
> "I'm citing Johnson 2022 for the claim that teacher efficacy affects student outcomes. Can you find the specific passage that supports this?"

**Visual Inspection**
> "Show me the figures from Chapter 3 of the Williams dissertation."
### The Thesis
Hey! Glad you made it this far down in the `README.md` :) I know this is a lot of text, but for the folks that care, here's the "why" behind this tool. My thesis on the state of AI tools right now and where I'd like this to go. 

ChatGPT was released in November of 2022 and by in large introduced Large Language Models (LLMs) to the world. Quickly, it swept through industry and academia much to many's delight and many other's horror. It's unequivocally changed the default mode in which people approach problems. As someone who was very much programming both before and after Cursor took the world by storm, I can attest, my workflow for creating code has dramatically changed--mostly for the better (While this is not entirely true, the full tangent here must be left for another day unfortunately).

While of course I can discuss the ways in which AI has improved and drastically changed many legacy workflows, it's also important to note many of its negative effects. It has greatly increased the risk of cognitive off loading during learning. The resource consumption of training and using of these LLMs is incredibly high. And, LLMs have been a marred with inconsistency, unexplainability and straight up hallucinations. So, when you look to scale their usage in any "real" setting, you have to accept the inevitability of failure. But still LLMs are incredibly powerful tools that represent the bleeding edge of where NLP stands today, **so why aren't they being used meaningfully**?

And, yes, I do understand tools like Elicit, Scite, and the million others that are competing for advertising space on my google results page exist. But, for me, they represent a black box. Their websites are just long eloquent "Trust Me Bros" that they've indexed petabytes of academic papers across uncountable academic disciplines to *leading experts* satisfaction. To me, this seems an impossible task. As someone who's worked with field experts to build out these system before and who believes that these models are just non-deterministic probabilistic black boxes and not tiny pieces of a god yet discovered, the whole task for building these systems comes down to single task: prompt engineering. 

> The act of iteratively designing a prompt on a set of inputs and outputs on a specific model

It's weird though, while I could try and design a tool that does this automatically, I don't think it would be as good as these experts. My argument is that the best person to have within that drivers seat, technician is the *expert themselves* as they will have the most precise and rigioursy content knowledge to check and refine these interactions.

While the tool above is just a small step towards this, it is my goal to begin to fill this gap.

All the best,
Elliot
