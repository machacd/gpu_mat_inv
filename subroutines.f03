module subroutines

contains
subroutine write_matrix(matrix)


real :: matrix(:,:)
integer :: i

do i=1,size(matrix,dim=1)
  	write (*,*) matrix(i,:)
    write(*,*) ""
end do

end subroutine

subroutine inv_mat(matrix,art) 
INTEGER :: i,m
real :: matrix(:,:)
real,ALLOCATABLE :: matrix_aug(:,:)
character(len=3) :: art
INTEGER,ALLOCATABLE :: ipiv(:)
real,ALLOCATABLE :: work(:)
INTEGER :: info=0
m=size(matrix,1)
if (art .eq. 'cpu') then
i =INT(m)
IF (ALLOCATED(work)) THEN
 DEALLOCATE(work)
ENDIF
ALLOCATE(work(i))
work=0
IF (ALLOCATED(ipiv)) THEN
 DEALLOCATE(ipiv)
ENDIF
ALLOCATE(ipiv(i))
ipiv=0

CALL sgetrf(i,i,matrix,i,ipiv,work,info)
IF ( info .NE. 0 ) THEN
 write(*,*) 'not OK'
END IF
CALL sgetri(i,matrix,i,ipiv,work,i,info)
IF ( info .NE. 0 ) THEN
 write(*,*) 'not OK'
END IF
else if (art .eq. 'gpu') then
  IF (ALLOCATED(matrix_aug)) THEN
    DEALLOCATE(matrix_aug)
  ENDIF
  ALLOCATE(matrix_aug(m,2*m))
  matrix_aug(:,1:m)=matrix
  matrix_aug(:,m+1:2*m)=0
  do i=1,m
    matrix_aug(i,i+m)=1
  end do
  call kernel_wrapper(matrix_aug,m)
! call write_matrix(matrix_aug)
  matrix=matrix_aug(:,m+1:2*m)
else
  write (*,*) "Unknown type of matrix inversion"
end if


END subroutine inv_mat

end module
