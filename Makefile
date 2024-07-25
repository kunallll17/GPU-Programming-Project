CUDA_PATH ?= /usr/local/cuda

INCLUDES = -I$(CUDA_PATH)/include
LIBS = -L$(CUDA_PATH)/lib64 -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

all: image_processor

image_processor: src/main.cu src/image_processing.cu
	$(CUDA_PATH)/bin/nvcc -o image_processor src/main.cu src/image_processing.cu $(INCLUDES) $(LIBS)

clean:
	rm -f image_processor
