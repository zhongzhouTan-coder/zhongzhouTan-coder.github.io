---
kind: web-extraction
source_url: "https://mathworld.wolfram.com/KroneckerProduct.html"
final_url: "https://mathworld.wolfram.com/KroneckerProduct.html"
canonical_url: "https://mathworld.wolfram.com/KroneckerProduct.html"
title: "Kronecker Product -- from Wolfram MathWorld"
author: ""
published_at: ""
captured_at: "2026-08-04T12:28:35.676Z"
content_sha256: 41ef5673a67ab59fd862ac4e1685d4b022b4ef5a6f4ce93946f490e496dbb6d1
renderer: http
extractor: "defuddle@0.13.0 + turndown@7.2.4"
---

Given an [matrix](https://mathworld.wolfram.com/Matrix.html) and a [matrix](https://mathworld.wolfram.com/Matrix.html) , their Kronecker product , also called their matrix direct product, is an [matrix](https://mathworld.wolfram.com/Matrix.html) with elements defined by

(1)

where

(2)

(3)

For example, the matrix direct product of the [matrix](https://mathworld.wolfram.com/Matrix.html) and the [matrix](https://mathworld.wolfram.com/Matrix.html) is given by the following [matrix](https://mathworld.wolfram.com/Matrix.html),

![\[a_(11)B a_(12)B; a_(21)B a_(22)B\]](https://mathworld.wolfram.com/images/equations/KroneckerProduct/Inline20.svg)

(4)

![\[a_(11)b_(11) a_(11)b_(12) a_(12)b_(11) a_(12)b_(12); a_(11)b_(21) a_(11)b_(22) a_(12)b_(21) a_(12)b_(22); a_(11)b_(31) a_(11)b_(32) a_(12)b_(31) a_(12)b_(32); a_(21)b_(11) a_(21)b_(12) a_(22)b_(11) a_(22)b_(12); a_(21)b_(21) a_(21)b_(2...](https://mathworld.wolfram.com/images/equations/KroneckerProduct/Inline23.svg)

(5)

The matrix direct product is implemented in the [Wolfram Language](http://www.wolfram.com/language/) as [KroneckerProduct](http://reference.wolfram.com/language/ref/KroneckerProduct.html) \[*a*, *b*\].

The matrix direct product gives the [matrix](https://mathworld.wolfram.com/Matrix.html) of the [linear transformation](https://mathworld.wolfram.com/LinearTransformation.html) induced by the [vector space tensor product](https://mathworld.wolfram.com/VectorSpaceTensorProduct.html) of the original [vector spaces](https://mathworld.wolfram.com/VectorSpace.html). More precisely, suppose that

(6)

and

(7)

are given by and . Then

(8)

is determined by

(9)
