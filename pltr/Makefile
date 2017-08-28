# A Makefile for compilation

JC=javac
JV=java
OUTDIR=classes

all:
	@echo "Make sure to use java 1.7 ..."
	@$(JC) -Xlint -d $(OUTDIR) -Xlint src/pltr/*.java
	@echo "Built into directory:" $(OUTDIR)
	@# @echo "Running Test ..."
	@# @cd $(OUTDIR) && $(JV) -Xmx8G pltr.Tester
	
clean:
	@if [ -d "classes/pltr" ]; then rm -r classes/*; fi
	
