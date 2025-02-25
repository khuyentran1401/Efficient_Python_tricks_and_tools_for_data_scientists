update: build publish
build:
	jb build .
publish:
	ghp-import -n -p -f _build/html
