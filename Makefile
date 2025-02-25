update: build copy git_add
build: 
	jb build .
copy: 
	rm -rf docs/*
	cp -r _build/html/* docs
git_add:
	git add .
	git status
