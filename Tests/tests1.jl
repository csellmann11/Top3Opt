using LinearAlgebra






A = rand(10,10)

w = rand(10)


W = diagm(w)


WA = W*A 

wA = w.*A 


wA == WA


AW = A*W 

Aw = w'.*A 

Aw == AW