# Lit Lake: Search & Analyze Literature Libraries with Claude
Connect [Claude](https://www.anthropic.com/claude) (and others) to [Zotero](https://www.zotero.org/) with one click. Discover related papers through semantic search, analyze references with full context and allow Claude to access the full text (including embedded images) of your library.

## Installation
### Claude (recommended)
Claude is the easiest and fastest way to get set up with Lit Lake. If you do not have Claude installed on your computer, download it [here](https://claude.com/download).

> If you're interested in learning more about what an `.mcpb` file is, read Anthropic's blog post [here](https://www.anthropic.com/engineering/desktop-extensions)!

#### Via MCPB
1. If you do not have Claude desktop installed, install it [here](https://claude.com/download)
2. Click [here](ttps://github.com/ElliotRoe/lit-lake/releases/latest) to view the latest version of Lit Lake
3. Click the file that ends with `.mcpb`
4. Wait for it to download
5. Double click the downloaded file
6. Click 'Install'

> Note: If double clicking does not immediately bring you to the installation screen within Claude, go to Settings > Extensions > Advanced settings > Install Extension

### Other LLMs
To use Lit Lake with other LLM clients (like LM Studio, Cherry studio, etc), you'll just need to download the binary file and make it executable then configure it globally. Honestly, I haven't configured it yet with another client, if you are attempting, please reach out and I can help, then I'll add the instructions back here.
## Use Cases
#### Paper Discovery & Search 
The search ability is great for discovering individual or groups of papers based on a topic. For every paper, depending on what is available in Zotero, Claude can search for keywords and search semantically within the titles and abstracts of your references. Notably, due to how I've implemented this tool, Claude also has the ability to combine these to two types of search in arbitrary and programatic ways, for more on this, see the advanced example.

**Basic Keyword Search Example** 
> "Can you find all articles in my library that have the keyword 'pedagogical content knowledge' or 'PCK' within their abstract?"

**Basic Example** 
> "Can you find all articles in my library that have the keyword 'pedagogical content knowledge' or 'PCK' within their abstract?"
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
