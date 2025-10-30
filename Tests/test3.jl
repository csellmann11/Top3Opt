using StaticArrays

println("Hello, World!")

A = rand(10,10)
# write A to file and close file
open("A.txt", "w") do file
    write(file, string(A))
end