cmake_minimum_required(VERSION 3.8)

project(HelloWorld C)
add_executable(HelloWorld main.c)
set_property(TARGET HelloWorld PROPERTY LINK_FLAGS "-lm")
