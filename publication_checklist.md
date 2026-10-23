# Publication checklist

## Before sharing
- [ ] Run `python3 run_all.py` and `python3 -m unittest discover -s tests -v` on the target Mac.
- [ ] Inspect SVGs/CSVs and confirm the article's numbers match `outputs/metrics_summary.csv`.
- [ ] Keep the synthetic-simulator limitation prominent; do not call this a real-LLM evaluation or safety proof.
- [ ] Record Python version, package version, seeds and any modifications.

## GitHub
- [ ] Add the folder to a repository or create a new repository; include code, tests, data, outputs, licences, citation metadata and references.
- [ ] Verify the public repository and raw CSV/SVG files open after pushing.
- [ ] Do not claim a DOI unless one has actually been minted.

## Website
- [ ] Adapt `article_draft.md` to the existing Masmatix Insights HTML template; verify canonical URL, author, date, title and internal links.
- [ ] Copy SVGs into the website image directory and check relative paths from the article URL.
- [ ] Add one card to `insights/index.html` and one URL entry to `sitemap.xml`, without duplicates.
- [ ] Run `git diff --check`, commit and push; then open the live article and verify every graph and GitHub link.
- [ ] A successful Git push alone does not prove the live page renders correctly.

## Google Search Console
- [ ] If the site is verified in Search Console, inspect the canonical live URL and request indexing if appropriate.
- [ ] Indexing requests do not guarantee crawling, indexing, ranking or visibility. Google indexing is not certification or endorsement.

## Kaggle (optional)
- [ ] Upload `data/agent_messages.csv`, `outputs/metrics_summary.csv`, `outputs/seed_summary.csv`, `DATA_CARD.md`, `DATA_LICENSE.txt` and the relevant code/README.
- [ ] Use a title such as “Synthetic Agent Uncertainty Handoff Benchmark”. State clearly that all rows are synthetic.
- [ ] Include generation command, seed list, split definition, metric definitions and limitations.
- [ ] Keep the generated-data CC BY 4.0 licence separate from the code's MIT licence.
- [ ] Run the Kaggle notebook before publishing; do not imply university endorsement or real-world validation.

## Archiving
- [ ] After review, consider a versioned release and Zenodo, OSF or Figshare for a DOI.
- [ ] Verify the archive record and licence before citing it.
