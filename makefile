LIBS = -llapack -L /opt/cuda/lib64 -I /opt/cuda/include -lcudart -lcuda -lcublas
FLAGS = -c -O3
CCOMPILER = /opt/cuda/bin/nvcc
build: subroutines.o main.o cudatest.o
	gfortran -o main.a $^ $(LIBS)
main.o: main.f03
	gfortran $(FLAGS) $<
subroutines.o: subroutines.f03
	gfortran $(FLAGS) $<
cudatest.o: cudatest.cu 
	$(CCOMPILER) $(FLAGS) $<
clean:
	rm *.o *.mod
