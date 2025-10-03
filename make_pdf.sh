#!/bin/bash

# Quick PDF generation script for TIC

echo "Generating TIC PDF for arXiv submission..."

# Option 1: If you have pandoc
if command -v pandoc &> /dev/null; then
    pandoc TEMPO_manuscript.md -o TIC.pdf --pdf-engine=xelatex
    echo "PDF created using pandoc: TIC.pdf"
else
    echo "Pandoc not found. Please use one of these alternatives:"
    echo "1. Copy TEMPO_manuscript.md to https://md2pdf.netlify.app/"
    echo "2. Open TEMPO_manuscript.md in VS Code with Markdown PDF extension"
    echo "3. Paste into Google Docs and export as PDF"
fi

echo ""
echo "Don't forget to:"
echo "1. Open figure1.html in browser and screenshot it"
echo "2. Insert the figure after Section 2.2 in the PDF"
echo "3. Submit to arXiv category: q-bio.NC"
echo ""
echo "GitHub repository ready at: https://github.com/WAR-0/TIC"