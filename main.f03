! This program compares two methods for matrix inversion.
! The compared quantities are the time and the accuracy of some
! random element.
program fortran_cuda_playground
use subroutines
implicit none

real,allocatable :: A(:,:)
real,allocatable :: A_cpu(:,:)
real :: x
integer,parameter :: dim=2048
integer :: i,j
real :: cpu_time_start, cpu_time_stop

! generate matrices
allocate(A(dim,dim))
allocate(A_cpu(dim,dim))
A=0
do i=1,dim
  do j=1,dim
    call random_number(x)
    A(i,j)=x
    A_cpu(i,j)=x
  end do
end do
! do the inversion with lapack and measure the time
call cpu_time(cpu_time_start)
call inv_mat(A,'gpu')
call cpu_time(cpu_time_stop)
write(*,*) dim
write(*,*) "GPU time:"
write(*,*) cpu_time_stop- cpu_time_start
write(*,*) "GPU result:"
write(*,*) A(3,4)
! do the inversion with the GPU
call cpu_time(cpu_time_start)
call inv_mat(A_cpu,'cpu')
call cpu_time(cpu_time_stop)
write(*,*) "CPU time:"
write(*,*) cpu_time_stop- cpu_time_start
write(*,*) "CPU result:"
write(*,*) A_cpu(3,4)



end program
