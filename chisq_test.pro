pro chisq_test

ntime = 100000L
nmodel = 32L
model = dindgen(ntime*nmodel)
model = model mod ntime
obs = model[0:ntime-1] + 1.*randomn(42,ntime)
sigma = 1.+0.*dindgen(ntime)
chisq = dblarr(nmodel)
chisq_idl = dblarr(nmodel)

t0 = systime(/seconds)
t = call_external(getenv('EXOFAST_PATH') + '/exofast_cuda.so',$
                  'calc_chisq_cuda',model,obs,sigma,ntime,nmodel,chisq,VALUE=[0,0,0,1,1,0],/D_VALUE)
tcuda = systime(/seconds)-t0
print, 'Time for CUDA: ' + strtrim(tcuda,2)

t0 = systime(/seconds)
chisq_idl[0] = total(((model[0:ntime-1]-obs[0:ntime-1])/sigma[0:ntime-1])^2)
chisq_idl[1] = total(((model[ntime:2*ntime-1]-obs[0:ntime-1])/sigma[0:ntime-1])^2)
tidl = (systime(/seconds)-t0)*nmodel/2.0
print, 'Time for IDL : ' + strtrim(tidl,2)

print, 'Difference between IDL and CUDA: model 0: ' + strtrim(max(abs(chisq_idl[0]-chisq[0])),2)

end

