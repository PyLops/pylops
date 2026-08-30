# PyLops AI Policy

**Table of Contents**

- [The Short Version](#the-short-version)
- [The Longer Version](#the-longer-version)
  - [Philosophy](#philosophy)
  - [Coding Agents](#coding-agents)


## The Short Version

Use AI like you used Google and StackOverflow. Own the final solution like you owned it before.

However, since AI still behaves differently from humans (it is undoubtedly better at certain tasks and worse at others), always acknowledge directly - e.g., let Claude Code author a PR that was strongly driven by it - or indirectly - e.g., explain in the PR where and how AI was used. This helps reviewers and maintainers to pay attention in different ways when reviewing a human-driven code vs an AI-driven code.

## The Longer Version

### Philosophy

PyLops has always been a forward-thinking and inclusive project. At a time when Python was largely regarded as a scripting language for mundane data manipulation and for stitching together HPC software written in lower-level programming languages, we believed that solving large-scale inverse problems efficiently and scalably didn’t necessarily require reaching for those languages. Instead, we focused on achieving greater expressivity, stronger abstractions, and ease of use—all while retaining the power and performance needed for demanding applications.

With the emergence of Coding Agents, the way we approach software development is changing, and we do not intend to be one of those communities that buries its head in the sand and carries on with business as usual. We instead encourage everyone to experiment with AI and Coding Agents and benefit from them in all stages of development.

This however does not mean that we encourage our developers (especially newcomers) to vibe-code complex solutions with little to no control on the physical outcome - lines of code! AI and Coding Agents should be treated as colleagues during pair-coding sessions: they can help in the ideation phase, during development, and in later stages to ensure consistency and act as additional attentive reviewers.

The only **strong recommendation** that we provide to anyone contributing code to PyLops is to be transparent about their use of AI/Coding Agents. We must recognize that AI still behaves differently from humans - it is undoubtedly better than us in certain tasks but it is still worse in other tasks. If we, reviewers and maintainers, know how a piece of code was generated, we can approach the review process slightly differently whether we review a human-driven code vs an AI-driven code.

### Coding Agents

In order to help our developers, we are committed to provide some of basic ingredients that allow Coding Agent to perform at their best. We aim to be as much as possible vendor-agnostic, and therefore we will provide equivalent versions of *.md* files, skills/commands, etc. that are suitable for one or another Coding Agent.

More specifically, we currently provide:

- ``AGENTS.md / CLAUDE.md``: basic set of instructions that tell coding agents how to work with our specific software project.

- ``.pi/prompts/optest.md`` / ``.claude/skills/optest``: a skill to increase the test coverage of an operator;

- ``.claude/skills/newop``: a skill to create a new operator from a mathematical description or a plain implementation of forward and adjoint from file or URL;

🤖🤖 **This Policy was written by humans and polished by AI** 🤖🤖
