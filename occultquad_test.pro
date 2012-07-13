pro occultquad_test

ntime = 100000L
nmodel = 32L
z = dindgen(ntime*nmodel)
z = 2.0*(z mod ntime)/(ntime)
u1 = dblarr(nmodel)*0.+0.5
u2 = dblarr(nmodel)*0.+0.5
p = 0.2*dindgen(nmodel)/(nmodel)
muo1 = dblarr(ntime*nmodel)
mu0 = dblarr(ntime*nmodel)
muo1_idl = dblarr(ntime)
mu0_idl = dblarr(ntime)

t0 = systime(/seconds)
t = call_external(getenv('EXOFAST_PATH') + '/exofast_cuda.so',$
                  'occultquad_cuda',z,u1,u2,p,ntime,nmodel,muo1,mu0,VALUE=[0,0,0,0,1,1,0,0],/D_VALUE)
tcuda = systime(/seconds)-t0
print, 'Time for CUDA: ' + strtrim(tcuda,2)

u1tmp = u1[1]
u2tmp = u2[1]
ptmp = p[1]
t0 = systime(/seconds)
exofast_occultquad,z,u1tmp,u2tmp,ptmp,muo1_idl,mu0_idl
tidl = (systime(/seconds)-t0)*nmodel

;print, 'Time for CUDA: ' + strtrim(tcuda,2)
print, 'Time for IDL : ' + strtrim(tidl,2)
print, 'Max difference between IDL and CUDA: muo1 ' + strtrim(max(abs(muo1[nmodel:2*nmodel-1]-muo1_idl)),2)
print, 'Max difference between IDL and CUDA: mu0  ' + strtrim(max(abs(mu0[nmodel:2*nmodel-1]-mu0_idl)),2)

end


pro occultquad_only_test

ntime = 100000L
nmodel = 32L
z = dindgen(ntime*nmodel)
z = 2.0*(z mod ntime)/(ntime)
u1 = dblarr(nmodel)+0.5
u2 = dblarr(nmodel)+0.5
p = 0.2*dindgen(nmodel)/(nmodel)
muo1 = dblarr(ntime*nmodel)
muo1_idl = dblarr(ntime)

t0 = systime(/seconds)
t = call_external(getenv('EXOFAST_PATH') + '/exofast_cuda.so',$
                  'occultquad_only_cuda',z,u1,u2,p,ntime,nmodel,muo1,VALUE=[0,0,0,0,1,1,0],/D_VALUE)
tcuda = systime(/seconds)-t0

u1tmp = u1[1]
u2tmp = u2[1]
ptmp = p[1]
t0 = systime(/seconds)
exofast_occultquad,z,u1tmp,u2tmp,ptmp,muo1_idl,mu0_idl
tidl = (systime(/seconds)-t0)*nmodel

print, 'Time for CUDA: ' + strtrim(tcuda,2)
print, 'Time for IDL : ' + strtrim(tidl,2)
print, 'Max difference between IDL and CUDA: ' + strtrim(max(abs(muo1[nmodel:2*nmodel-1]-muo1_idl)),2)

end

pro occult_uniform_slow_test

ntime = 100000L
nmodel = 32L
z = dindgen(ntime*nmodel)
z = 2.0*(z mod ntime)/(ntime)
u1 = dblarr(nmodel)+0.5
u2 = dblarr(nmodel)+0.5
p = 0.2*dindgen(nmodel)/(nmodel)
mu0 = dblarr(ntime*nmodel)
mu0_idl = dblarr(ntime)

t0 = systime(/seconds)
t = call_external(getenv('EXOFAST_PATH') + '/exofast_cuda.so',$
                  'occult_uniform_slow_cuda',z,u1,u2,p,ntime,nmodel,mu0,VALUE=[0,0,0,0,1,1,0],/D_VALUE)
tcuda = systime(/seconds)-t0

u1tmp = u1[1]
u2tmp = u2[1]
ptmp = p[1]
t0 = systime(/seconds)
exofast_occultquad,z,u1tmp,u2tmp,ptmp,muo1_idl,mu0_idl
tidl = (systime(/seconds)-t0)*nmodel

print, 'Time for CUDA: ' + strtrim(tcuda,2)
print, 'Time for IDL : ' + strtrim(tidl,2)
print, 'Max difference between IDL and CUDA: ' + strtrim(max(abs(mu0[nmodel:2*nmodel-1]-mu0_idl)),2)

end


