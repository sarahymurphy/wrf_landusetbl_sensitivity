jupyter-book build ../wrf_analysis_book
git add *
git commit -m "pushed by script"
git push origin main
ghp-import -n -p -f _build/html