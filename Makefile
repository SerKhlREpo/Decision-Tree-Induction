all: funcDef.cpp funcDef.hpp main.cpp treeNode.cpp treeNode.hpp
	g++ -std=c++17 main.cpp funcDef.cpp treeNode.cpp -o main

clean: 
	rm -f main