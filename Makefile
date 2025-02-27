update: build git_add
clean: 
	jb clean . --all
build:
	jb build .
git_add:
	git add .
	git status
publish:
	ghp-import -n -p -f _build/html