build: 
	jb build .
copy: 
	cp -r _build/html/* docs
git_add:
	git add .
	git status
update: build copy git_add
