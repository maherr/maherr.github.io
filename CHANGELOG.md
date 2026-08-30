# Changelog

All notable releases of maherr.dev are recorded here. The product uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) and the release
history follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

**Last updated:** 2026-08-29 22:06:43 -0400

## [Unreleased]

Changes after the current named release will be recorded here.

## [1.0.0] - 2026-08-29

- First named release; prior history was unversioned.
- Added the product landing, task-oriented usage, exact reference, privacy,
  accessibility, troubleshooting, rollback, deployment, and feature-coverage
  documentation in `README.md`.
- Added an explicit `VERSION` authority and this Keep-a-Changelog history.
- Added a keyboard-operable `Theme` select with Auto (system), Light, Dark,
  and opt-in Dracula Classic choices. Explicit choices persist under
  `maherr-theme`; stale values fall back to the original system behavior
  without rewriting storage.
- Added semantic Dracula Classic tokens for the site shell, prose, links,
  code, borders, controls, selection, focus, and article-embedded SVG charts.
- Added an MIT source-code license boundary while keeping authored content
  under `content/` all rights reserved unless a file says otherwise.
- Versioned surface: templates, Sass, Zola configuration, static assets, feed,
  and GitHub Pages build procedure. Editorial post content remains outside
  independent software release numbering.

Release proof for this candidate: Zola 0.22.1 build, product-completeness
checker, focused theme-state test, rendered index and article checks in both
original system schemes plus Dracula Classic, persistence and stale-value
checks, keyboard/focus and overflow checks, computed contrast, and
deuteranomaly review at severities 0.6 and 0.8.
