#include <misc.h>
#include <params.h>

module cloudbrain_module
use shr_kind_mod,    only: r8 => shr_kind_r8
use ppgrid,          only: pcols, pver, pverp
use history,         only: outfld, addfld, add_default, phys_decomp
use nt_FunctionsModule, only: nt_tansig
use physconst,       only: gravit,cpair
implicit none

save

private                   ! Make default type private to the module
integer,parameter :: hiddenlayerSize = 2
integer,parameter :: outputlayerSize = 3
integer, parameter :: nbhiddenlayers = 1
integer,parameter :: inputlayerSize = 4

public cloudbrain

contains: 

subroutine define_neuralnet_SPDT (mu_in,std_in,weights_input, bias_input,bias_output,weights_output)
real(r8), intent(out) :: mu_in(inputlayerSize)
real(r8), intent(out) :: std_in(inputlayerSize)
real(r8), intent(out) :: bias_input(hiddenlayerSize)
real(r8), intent(out) :: bias_output(outputlayerSize)
real(r8), intent(out) :: weights_input(hiddenlayerSize, inputlayerSize)
real(r8), intent(out) :: weights_output(outputlayerSize,hiddenlayerSize)

mu_in(:) = (/-9.12332e-01 /)
std_in(:) = (/9.44444e-07 /)
bias_input_in(:) = (/8.314840e-02, -1.178977e-01 /)
bias_output(:) = (/-1.063119e-01, 1.999398e-01 /)
weights_input(0:) = (/7.489552e-02, -3.008047e-02, 4.961192e-02, 1.249248e-01 /)
weights_input(1:) = (/7.489552e-02, -3.008047e-02, 4.961192e-02, 1.249248e-01 /)
weights_output(0:) = (/-2.654571e-01, -6.213908e-01, 4.264377e-01, 5.954914e-01 /)
weights_output(1:) = (/-2.654571e-01, -6.213908e-01, 4.264377e-01, 5.954914e-01 /)
weights_output(2:) = (/-2.654571e-01, -6.213908e-01, 4.264377e-01, 5.954914e-01 /)


end subroutine define_neuralnet_SPDT


subroutine define_neuralnet_SPDQ (mu_in,std_in,weights_input, bias_input,bias_output,weights_output)
real(r8), intent(out) :: mu_in(inputlayerSize)
real(r8), intent(out) :: std_in(inputlayerSize)
real(r8), intent(out) :: bias_input(hiddenlayerSize)
real(r8), intent(out) :: bias_output(outputlayerSize)
real(r8), intent(out) :: weights_input(hiddenlayerSize, inputlayerSize)
real(r8), intent(out) :: weights_output(outputlayerSize,hiddenlayerSize)

mu_in(:) = (/-9.12332e-01 /)
std_in(:) = (/9.44444e-07 /)
bias_input_in(:) = (/8.314840e-02, -1.178977e-01 /)
bias_output(:) = (/-1.063119e-01, 1.999398e-01 /)
weights_input(0:) = (/7.489552e-02, -3.008047e-02, 4.961192e-02, 1.249248e-01 /)
weights_input(1:) = (/7.489552e-02, -3.008047e-02, 4.961192e-02, 1.249248e-01 /)
weights_output(0:) = (/-2.654571e-01, -6.213908e-01, 4.264377e-01, 5.954914e-01 /)
weights_output(1:) = (/-2.654571e-01, -6.213908e-01, 4.264377e-01, 5.954914e-01 /)
weights_output(2:) = (/-2.654571e-01, -6.213908e-01, 4.264377e-01, 5.954914e-01 /)


end subroutine define_neuralnet_SPDQ


end module cloudbrain
