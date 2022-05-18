
echo -n "What changes did you make?    "
set changes = $<
jupyter-book build ../wrf_analysis_book

git add *
git commit -m "$changes[1]"
git push origin main

ghp-import -n -p -f _build/html
