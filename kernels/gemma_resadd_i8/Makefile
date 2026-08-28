PROJECT = gemma_resadd_i8

MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

include ../common.mk

data: resadd.py
	python3 resadd.py

kernel.mu.o: data
