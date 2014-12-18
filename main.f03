program fortran_cuda_playground
use subroutines
implicit none

real,allocatable :: A(:,:)
real,allocatable :: A_cpu(:,:)
real :: x
integer,parameter :: dim=2048
integer :: i,j
real :: cpu_time_start, cpu_time_stop

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
!call write_matrix(A)
!call write_matrix(A_cpu)
call cpu_time(cpu_time_start)
call inv_mat(A,'gpu')
call cpu_time(cpu_time_stop)
write(*,*) dim
write(*,*) "GPU time:"
write(*,*) cpu_time_stop- cpu_time_start
write(*,*) "GPU result:"
write(*,*) A(3,4)
call cpu_time(cpu_time_start)
call inv_mat(A_cpu,'cpu')
call cpu_time(cpu_time_stop)
write(*,*) "CPU time:"
write(*,*) cpu_time_stop- cpu_time_start
write(*,*) "CPU result:"
write(*,*) A_cpu(3,4)

!call write_matrix(A)
!call write_matrix(A_cpu)


end program
