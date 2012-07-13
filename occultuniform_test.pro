pro occultuniform_test

ntime = 100000L
nmodel = 32L
z = dindgen(ntime*nmodel)
z = 2.0*(z mod ntime)/(ntime)
p = 0.2*dindgen(nmodel)/(nmodel)
mu0 = dblarr(ntime*nmodel)
mu0_idl = dblarr(ntime)

t0 = systime(/seconds)
t = call_external(getenv('EXOFAST_PATH') + '/exofast_cuda.so',$
                  'occult_uniform_cuda',z,p,ntime,nmodel,mu0,VALUE=[0,0,1,1,0],/D_VALUE)
tcuda = systime(/seconds)-t0

ptmp = p[1]
t0 = systime(/seconds)
u1tmp = 0.5
u2tmp = 0.5
exofast_occultquad,z,u1tmp,u2tmp,ptmp,muo1_idl,mu0_idl
tidl = (systime(/seconds)-t0)*nmodel

print, 'Time for CUDA: ' + strtrim(tcuda,2)
print, 'Time for IDL : ' + strtrim(tidl,2)
print, 'Max difference between IDL and CUDA: mu0  ' + strtrim(max(abs(mu0[nmodel:2*nmodel-1]-mu0_idl)),2)

end



