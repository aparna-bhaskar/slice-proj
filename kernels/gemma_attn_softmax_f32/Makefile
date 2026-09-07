PROJECT = gemma_attn_softmax_f32

MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

PYTHON_REF ?= python3
GEMMINI_SW_PATH ?= $(RADIANCE_LIB_PATH)/mxgemmini

include ../../radiance-support/common.mk

data: softmax.py
	$(PYTHON_REF) softmax.py

kernel.mu.o: data
