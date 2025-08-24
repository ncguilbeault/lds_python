all: ldsInference.pdf ekfInference.pdf

ldsInference.pdf: ldsInference.tex derivationKalmanFilterEqs.tex
	pdflatex ldsInference.tex
	bibtex ldsInference.aux
	pdflatex ldsInference.tex
	pdflatex ldsInference.tex

%.pdf: %.tex
	pdflatex $<
	bibtex $*
	pdflatex $<
	pdflatex $<

