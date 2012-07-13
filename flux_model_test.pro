pro flux_model_test

ntime = 100000L
nmodel = 32L
time = dindgen(ntime)/(ntime)*365.25*3.5
tperi = dindgen(nmodel)/(nmodel)*10.
period = 10.0+0.1*dindgen(nmodel)/(nmodel)
aoverrstar = 20.0+0.5*dindgen(nmodel)/(nmodel)
inc = (90.0+1.0*dindgen(nmodel)/(nmodel))*!dpi/180.
ecc = 0.1+0.3*dindgen(nmodel)/(nmodel)
omega = 2d0*!dpi*dindgen(nmodel)/(nmodel)
rpoverrstar = 0.01+0.01*dindgen(nmodel)/(nmodel)
u1 = dblarr(nmodel)+0.5
u2 = dblarr(nmodel)+0.5
flux_model = dblarr(ntime*nmodel)

t0 = systime(/seconds)
t = call_external(getenv('EXOFAST_PATH') + '/exofast_cuda.so',$
                  'flux_model_cuda',time,tperi,period,aoverrstar,inc,ecc,omega,rpoverrstar,u1,u2,ntime,nmodel,flux_model,VALUE=[0,0,0,0,0,0,0,0,0,0,1,1,0],/D_VALUE)
tcuda = systime(/seconds)-t0

print, 'Time for CUDA: ' + strtrim(tcuda,2)
;print, 'Time for IDL : ' + strtrim(tidl,2)
;print, 'Max difference between IDL and CUDA:' + strtrim(max(abs(eccanom-eccanom2)),2)

end



pro flux_model_wltt_test

ntime = 100000L
nmodel = 32L
time = dindgen(ntime)/(ntime)*365.25*3.5
tperi = dindgen(nmodel)/(nmodel)*10.
period = 10.0+0.1*dindgen(nmodel)/(nmodel)
aoverrstar = 20.0+0.5*dindgen(nmodel)/(nmodel)
ainau = aoverrstar*200.
inc = (90.0+1.0*dindgen(nmodel)/(nmodel))*!dpi/180.
ecc = 0.1+0.3*dindgen(nmodel)/(nmodel)
omega = 2d0*!dpi*dindgen(nmodel)/(nmodel)
rpoverrstar = 0.01+0.01*dindgen(nmodel)/(nmodel)
u1 = dblarr(nmodel)+0.5
u2 = dblarr(nmodel)+0.5
flux_model = dblarr(ntime*nmodel)

t0 = systime(/seconds)
t = call_external(getenv('EXOFAST_PATH') + '/exofast_cuda.so',$
                  'flux_model_wltt_cuda',time,tperi,period,aoverrstar,ainau,inc,ecc,omega,rpoverrstar,u1,u2,ntime,nmodel,flux_model,VALUE=[0,0,0,0,0,0,0,0,0,0,0,1,1,0],/D_VALUE)
tcuda = systime(/seconds)-t0

print, 'Time for CUDA: ' + strtrim(tcuda,2)
;print, 'Time for IDL : ' + strtrim(tidl,2)
;print, 'Max difference between IDL and CUDA:' + strtrim(max(abs(eccanom-eccanom2)),2)

end

pro chisq_flux_model_test

ntime = 100000L
nmodel = 32L
time = dindgen(ntime)/(ntime)*365.25*3.5
obs = 0.+0.*dindgen(ntime)/(ntime)
sigma = 0.001+0.*dindgen(ntime)/(ntime)*365.25*3.5
tperi = dindgen(nmodel)/(nmodel)*10.
period = 10.0+0.1*dindgen(nmodel)/(nmodel)
aoverrstar = 20.0+0.5*dindgen(nmodel)/(nmodel)
inc = (90.0+1.0*dindgen(nmodel)/(nmodel))*!dpi/180.
ecc = 0.1+0.3*dindgen(nmodel)/(nmodel)
omega = 2d0*!dpi*dindgen(nmodel)/(nmodel)
rpoverrstar = 0.01+0.01*dindgen(nmodel)/(nmodel)
u1 = dblarr(nmodel)+0.5
u2 = dblarr(nmodel)+0.5
;flux_model = dblarr(ntime*nmodel)
chisq = dblarr(ntime*nmodel)

t0 = systime(/seconds)
t = call_external(getenv('EXOFAST_PATH') + '/exofast_cuda.so',$
                  'chisq_flux_model_cuda',time,obs,sigma,tperi,period,aoverrstar,inc,ecc,omega,rpoverrstar,u1,u2,ntime,nmodel,chisq,VALUE=[0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],/D_VALUE)
tcuda = systime(/seconds)-t0

print, 'Time for CUDA: ' + strtrim(tcuda,2)
;print, 'Time for IDL : ' + strtrim(tidl,2)
;print, 'Max difference between IDL and CUDA:' + strtrim(max(abs(eccanom-eccanom2)),2)

end

pro chisq_flux_model_wltt_test

ntime = 100000L
nmodel = 32L
time = dindgen(ntime)/(ntime)*365.25*3.5
obs = 0.+0.*dindgen(ntime)/(ntime)
sigma = 0.001+0.*dindgen(ntime)/(ntime)*365.25*3.5
tperi = dindgen(nmodel)/(nmodel)*10.
period = 10.0+0.1*dindgen(nmodel)/(nmodel)
aoverrstar = 20.0+0.5*dindgen(nmodel)/(nmodel)
ainau = aoverrstar*200.
inc = (90.0+1.0*dindgen(nmodel)/(nmodel))*!dpi/180.
ecc = 0.1+0.3*dindgen(nmodel)/(nmodel)
omega = 2d0*!dpi*dindgen(nmodel)/(nmodel)
rpoverrstar = 0.01+0.01*dindgen(nmodel)/(nmodel)
u1 = dblarr(nmodel)+0.5
u2 = dblarr(nmodel)+0.5
;flux_model = dblarr(ntime*nmodel)
chisq = dblarr(ntime*nmodel)

t0 = systime(/seconds)
t = call_external(getenv('EXOFAST_PATH') + '/exofast_cuda.so',$
                  'chisq_flux_model_wltt_cuda',time,obs,sigma,tperi,period,aoverrstar,ainau,inc,ecc,omega,rpoverrstar,u1,u2,ntime,nmodel,chisq,VALUE=[0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],/D_VALUE)
tcuda = systime(/seconds)-t0

print, 'Time for CUDA: ' + strtrim(tcuda,2)
;print, 'Time for IDL : ' + strtrim(tidl,2)
;print, 'Max difference between IDL and CUDA:' + strtrim(max(abs(eccanom-eccanom2)),2)

end

