# Publishing checklist

Complete these items before making the GitHub repository public:

- [ ] Replace the provisional author entry in `CITATION.cff`.
- [ ] Add the accepted venue, year, DOI, paper URL, and final BibTeX.
- [ ] Select an explicit license for the original STAR-CVI code.
- [ ] Confirm that all modified CLIP files retain the OpenAI MIT notice.
- [ ] Confirm compatibility with the OpenGait academic-use terms.
- [ ] Decide whether the manuscript PDF may be public during peer review.
- [ ] Publish the exact dataset split only if AG-VPReID terms allow it.
- [ ] Add checkpoint links and SHA-256 checksums after upload.
- [ ] Run `python scripts/validate_repo.py` and `pytest -q`.
- [ ] Run a clean installation in a new OpenGait checkout.
- [ ] Complete at least one forward/backward smoke test on a CUDA GPU.
- [ ] Re-run the full 40k schedule and archive evaluation logs.
- [ ] Replace placeholder repository URLs in release metadata, if any.
- [ ] Review Git history for datasets, credentials, private paths, and large files.

