#!/bin/sh

# TEX_COMMAND=latex
TEX_COMMAND=xelatex

echo "****************** compile phase 1 ******************"
echo " "
${TEX_COMMAND} $1.tex

echo " "
echo "****************** compile phase 2 ******************"
echo " "
#makeindex $1.nlo -s nomencl.ist -o $1.nls

echo " "
echo "****************** compile phase 3 ******************"
echo " "
${TEX_COMMAND} $1.tex

echo " "
echo "****************** compile phase 4 ******************"
echo " "
bibtex $1.aux

echo " "
echo "****************** compile phase 5 ******************"
echo " "
${TEX_COMMAND} $1.tex

echo " "
echo "****************** compile phase 6 ******************"
echo " "
${TEX_COMMAND} $1.tex

echo " "
echo "****************** compile phase 7 ******************"
echo " "
dvipdf $1.dvi

echo "compilation done."

