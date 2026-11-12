# maherr.dev

**Last updated:** 2026-11-11 19:22:01 -0500

`maherr.dev` is a small Zola/GitHub Pages site for engineering notes. The
published site is at [maherr.dev](https://maherr.dev). It has no server,
database, analytics, cookies, or runtime dependency.

**Coverage:** Covers v1.0.0; feature sweep 2026-08-29 from the built index and article routes, theme selector, Atom feed, favicon, and GitHub Pages workflow.

## Open and build locally

Install [Zola 0.22.1](https://www.getzola.org/documentation/getting-started/installation/),
then run these commands from the repository root:

```sh
zola serve
```

The preview binds to `127.0.0.1:1111` by default. Open
`http://127.0.0.1:1111/` for the index or
`http://127.0.0.1:1111/rdna4-missing-rung/` and `http://127.0.0.1:1111/witness/` for the articles. A production-like
static build is generated with:

```sh
zola build
```

The generated `public/` directory is ignored by Git. Rebuild it after source
changes instead of editing generated HTML or CSS directly.

## Use the site

The header's `Theme` select is present on the index and article pages. It is a
native keyboard-operable control with four text-labelled choices:

- `Auto (system)` is the default and follows the browser's
  `prefers-color-scheme` setting.
- `Light` and `Dark` explicitly select the site's existing light and dark token
  sets.
- `Dracula Classic` applies the official Dracula palette through semantic site
  tokens and repaints the article's embedded charts in the same document.

Only explicit `light`, `dark`, and `dracula` choices are stored in
`localStorage` under the key `maherr-theme`. Choosing `Auto (system)` removes
that key. An unknown or stale stored value is ignored, left untouched, and
falls back to the original system behavior. The selection repaints without a
reload. Focus is visible with an outline, and the named choices remain
understandable without color.

## Product surface and documentation map

The versioned product surface is the site shell, templates, stylesheet, Zola
configuration, static assets, build, and feed generation. The Markdown under
`content/` is authored editorial content and is not an independently released
software surface. It remains all rights reserved unless an individual content
file states otherwise.

| Surface | Source or route | Behavior | Reference |
| --- | --- | --- | --- |
| Post index | `content/_index.md`, `/` | Lists published posts in date order. | This README, [index route](https://maherr.dev/) |
| Engineering article | `content/rdna4-missing-rung.md`, `/rdna4-missing-rung/` | Renders the Witness article, tables, code, and labelled SVG charts. | This README, [article route](https://maherr.dev/rdna4-missing-rung/) |
| Engineering article | `content/witness.md`, `/witness/` | Renders the Witness introduction: install, GPU matrix, speed and accuracy tables. | This README, [article route](https://maherr.dev/witness/) |
| Theme control | `templates/base.html`, `#theme-select` | Applies Auto, Light, Dark, or Dracula Classic in place. | "Use the site" above |
| Theme persistence | `templates/base.html`, `maherr-theme` | Stores only explicit theme choices; stale values fail open to system mode. | "Use the site" above |
| Semantic theme tokens | `sass/style.scss` | Paints text, links, code, borders, selection, focus, controls, and article SVG classes. | `sass/style.scss` |
| Atom feed | `/atom.xml` | Publishes the generated Atom feed for the site's posts. | `config.toml`, [feed](https://maherr.dev/atom.xml) |
| Favicon and domain | `static/favicon.svg`, `static/CNAME` | Serves the site mark and maps GitHub Pages to `maherr.dev`. | `static/` |
| Build and deploy | `zola build`, `.github/workflows/deploy.yml` | Builds with Zola 0.22.1 and deploys `public/` through GitHub Pages on `main` pushes or manual dispatch. | "Build and deploy" below |

There are fewer than three user-facing documentation files, so this README is
the tutorial, how-to, reference, explanation, troubleshooting, and operations
home rather than adding a separate documentation index.

## Configuration

`config.toml` is the only site configuration file. It defines the public base
URL, title, description, author handle, Sass compilation, feed filename, and
Markdown options. The theme choice is intentionally user-local browser state,
not Zola configuration.

The site has no API, CLI, import/export format, user account, server-side data,
or optional provider. The only browser-written value is the documented theme
key. Page content and generated feed data come from the checked-in `content/`
files. GitHub Pages receives the generated static artifact through the
workflow; the workflow does not run against personal data.

## Build and deploy

`.github/workflows/deploy.yml` is the canonical deployment procedure. It
installs Zola 0.22.1, runs `zola build`, uploads `public/`, and deploys through
GitHub Pages. The workflow runs for pushes to `main` or a manual
`workflow_dispatch`. `static/CNAME` supplies the custom domain.

Before a landing or deployment review, run the local build and inspect the
generated index, article, feed, favicon, and `CNAME` files. A source-only edit
does not change the public site until the workflow deploys a new build.

## Troubleshooting and rollback

### The theme does not change

Confirm that the page contains the `Theme` select, then choose a different
named option. If the browser has a stale value, open its storage inspector for
the site origin and inspect only the `maherr-theme` key. Delete that one key or
choose `Auto (system)` to return to the original system behavior. A stale value
should be ignored rather than rewritten by the site.

### A local page or style is missing

Run `zola build` or restart `zola serve` from the repository root. Check that
the requested source exists under `content/` and that the generated path is
under `public/`. Do not fix generated output by hand.

### A published change is not visible

Check the GitHub Pages workflow's build and deploy jobs, then hard-reload the
published origin. Verify that the served stylesheet hash changes after a
successful build. The release workflow, rather than a local preview, is the
authority for the public artifact.

### Roll back a landing

The repository's prior Git commit is the byte-level recovery point for this
Git-backed product. Restore the reviewed previous source, run `zola build`, and
let the normal GitHub Pages workflow deploy it. Rollback does not require a
database migration because the only persisted browser value is an independent
theme preference; choosing `Auto (system)` removes it.

## License boundary

`LICENSE` covers the site's source code and build configuration under the MIT
License. The editorial prose and other content under `content/` remain all
rights reserved unless a content file states otherwise. This boundary keeps
the reusable implementation license separate from the authored posts.
