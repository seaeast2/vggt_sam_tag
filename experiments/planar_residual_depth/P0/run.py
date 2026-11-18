"""P0: analytic finite planes, camera-Z depths, observation-only plane refinement."""
import json, hashlib, shutil, sys, platform, subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
import scipy
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
K = np.array([[240.,0,159.5],[0,240.,119.5],[0,0,1]])
H,W = 240,320

def dump(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def project(X,R,C):
    Y=(X-C)@R.T
    h=Y@K.T
    return h[:,:2]/h[:,2:],Y[:,2]

def rays(uv,R):
    return np.c_[uv,np.ones(len(uv))]@np.linalg.inv(K).T@R

def depth(p,uv,R,C):
    # p.X + 1 = 0; p=n/e. No ray normalization: depth is camera-Z.
    return -(1+C@p)/(rays(uv,R)@p)

def angle(p,q):
    return float(np.degrees(np.arccos(np.clip(abs(p@q)/(np.linalg.norm(p)*np.linalg.norm(q)),0,1))))

def fit_plane(X):
    c=X.mean(0)
    _,_,v=np.linalg.svd(X-c,full_matrices=False)
    n=v[-1]; e=-n@c
    return n/e

def residual(p,observations,cameras):
    out=[]
    for uv,vv,i,j in observations:
        Ri,Ci=cameras[i]; Rj,Cj=cameras[j]
        z=depth(p,uv,Ri,Ci)
        X=Ci+z[:,None]*rays(uv,Ri)
        pred,zj=project(X,Rj,Cj)
        rr=pred-vv
        bad=(z<=0)|(zj<=0)|(~np.isfinite(rr).all(1))
        rr=np.nan_to_num(rr,nan=1e6,posinf=1e6,neginf=-1e6)
        rr[bad]=1e6
        out.extend(rr.ravel())
    return np.array(out)

def analytic_jacobian(p,obs,cameras):
    blocks=[]
    for uv,vv,i,j in obs:
        Ri,Ci=cameras[i]; Rj,Cj=cameras[j]
        q=rays(uv,Ri); den=q@p; A=1+Ci@p
        z=-A/den
        dz=-(Ci[None,:]*den[:,None]-A*q)/den[:,None]**2
        X=Ci+z[:,None]*q
        h=((X-Cj)@Rj.T)@K.T
        M=K@Rj
        dp=(M[None,:2,:]*h[:,2,None,None]-h[:,:2,None]*M[None,2:3,:])/h[:,2,None,None]**2
        blocks.append(np.einsum('nij,nj,nk->nik',dp,q,dz).reshape(-1,3))
    return np.concatenate(blocks)

def optimize(p0,obs,cameras,fixed=False):
    # IRLS implements a radial 2D Huber loss (delta=1 pixel), same weights per point.
    a=np.ones(1) if fixed else p0.copy()
    weights=np.ones(len(residual(p0,obs,cameras))//2)
    unpack=lambda x:p0*x[0] if fixed else x
    for _ in range(20):
        sol=least_squares(lambda x: residual(unpack(x),obs,cameras)*np.repeat(np.sqrt(weights),2),a,
                          max_nfev=300,ftol=1e-12,xtol=1e-12,gtol=1e-12)
        a=sol.x
        norms=np.linalg.norm(residual(unpack(a),obs,cameras).reshape(-1,2),axis=1)
        new=1/np.maximum(norms,1.)
        if np.max(abs(new-weights))<1e-8: break
        weights=new
    return unpack(a),bool(sol.success),sol.nfev

def main():
    date=datetime.now().strftime('%Y%m%d')
    base=ROOT/'results/planar_residual_depth'
    seq=1
    while (base/f'synthetic-planes-P0-{date}-{seq:03}').exists(): seq+=1
    rid=f'synthetic-planes-P0-{date}-{seq:03}'
    out=base/rid; out.mkdir(parents=True,exist_ok=False)
    (out/'source').mkdir(); shutil.copy2(__file__,out/'source/run.py')
    (out/'logs').mkdir(); (out/'figures').mkdir()
    rng=np.random.default_rng(0)
    config={'seed':0,'size':[W,H],'K':K.tolist(),'baseline_m':0.3,'rotation_y_deg':5,
            'raw_normal_error_deg':12,'raw_offset_error_m':0.25,'huber_px':1,
            'thresholds':{'normal_deg':0.1,'relative_depth':0.001},
            'cases':['clean','tilt','offset','ripple','pure_rotation','collinear','outliers'],
            'plane_orientations_deg':[[0,0,0],[10,25,0],[65,0,0]],
            'scope':'CPU synthetic geometry only; no RGB rendering, SAM, VGGT or real data'}
    dump(out/'config_resolved.json',config)
    checks={}; rows=[]; example=None
    yy,xx=np.mgrid[:H,:W]; uvgrid=np.c_[xx.ravel(),yy.ravel()]
    for plane_id,angles in enumerate(config['plane_orientations_deg']):
        n=Rotation.from_euler('xyz',angles,degrees=True).apply([0.,0.,1.])
        center=np.array([0.,0.,3.]); e=-n@center; truth=n/e
        tangent=Rotation.from_euler('xyz',angles,degrees=True).as_matrix()[:,:2]
        for case in config['cases']:
            R1=Rotation.from_euler('y',5,degrees=True).as_matrix()
            C1=np.zeros(3) if case=='pure_rotation' else np.array([.3,0,0])
            cams=[(np.eye(3),np.zeros(3)),(R1,C1)]
            coords=rng.uniform(-.9,.9,(500,2))
            if case=='collinear': coords[:,1]=0
            X=center+coords@tangent.T
            ua,za=project(X,*cams[0]); ub,zb=project(X,*cams[1])
            valid=(za>0)&(zb>0)
            for u in (ua,ub): valid&=(u[:,0]>2)&(u[:,0]<W-3)&(u[:,1]>2)&(u[:,1]<H-3)
            ua,ub=ua[valid],ub[valid]
            order=rng.permutation(len(ua)); nf=int(.7*len(order)); fi,vi=order[:nf],order[nf:]
            observed=ub.copy()
            if case=='outliers':
                bad=fi[:max(1,len(fi)//5)]
                observed[bad]+=rng.uniform(-60,60,(len(bad),2))
            fit=[(ua[fi],observed[fi],0,1),(observed[fi],ua[fi],1,0)]
            hold=[(ua[vi],ub[vi],0,1),(ub[vi],ua[vi],1,0)]
            nraw=n.copy(); eraw=e
            if case in ('tilt','pure_rotation','collinear','outliers'):
                nraw=Rotation.from_rotvec(tangent[:,1]*np.radians(12)).apply(n)
                eraw=-nraw@center
            if case=='offset': eraw=e-.25
            praw=nraw/eraw
            normals=[]; allbase=[]; masks=[]; gt=[]
            for R,C in cams:
                z=depth(truth,uvgrid,R,C)
                P=C+z[:,None]*rays(uvgrid,R)
                local=(P-center)@tangent
                mask=(z>0)&(np.abs(local).max(1)<=1.)
                raw=depth(praw,uvgrid,R,C)
                if case=='ripple': raw+=.025*np.sin(uvgrid[:,0]/9)*np.cos(uvgrid[:,1]/11)
                # Fit on all raw plane pixels; no GT values are passed to this fit.
                normals.append(fit_plane(C+raw[mask,None]*rays(uvgrid[mask],R)))
                allbase.append(raw); masks.append(mask); gt.append(z)
            b1=fit_plane(np.concatenate([C+raw[m,None]*rays(uvgrid[m],R)
                         for (R,C),raw,m in zip(cams,allbase,masks)]))
            # Observability gate uses camera baseline and 2D spread, not truth normals.
            spread=np.linalg.svd(ua[fi]-ua[fi].mean(0),compute_uv=False)
            reason='PURE_ROTATION' if np.linalg.norm(C1)<1e-8 else ('COLLINEAR' if spread[-1]/spread[0]<1e-3 else None)
            b2,s2,_=optimize(b1,fit,cams,True)
            b3,s3,_=optimize(b1,fit,cams)
            eps=1e-6
            J=np.column_stack([(residual(b3+np.eye(3)[a]*eps,fit,cams)-residual(b3-np.eye(3)[a]*eps,fit,cams))/(2*eps) for a in range(3)])
            sv=np.linalg.svd(J,compute_uv=False)
            Ja=analytic_jacobian(b3,fit,cams)
            jac_error=float(np.max(abs(Ja-J))/max(1.,np.max(abs(Ja))))
            assert jac_error<1e-6, jac_error
            folder=out/'artifacts'/f'plane{plane_id}_{case}'
            dump(folder/'inputs/observations.json',{'fit_ids':fi.tolist(),'heldout_ids':vi.tolist(),
                'uv0':ua.tolist(),'uv1_observed':observed.tolist(),'cameras':[{'R':r.tolist(),'C':c.tolist()} for r,c in cams]})
            dump(folder/'inputs/truth.json',{'p':truth.tolist(),'finite_center':center.tolist(),'tangent':tangent.tolist(),'half_extent':1.})
            details={}
            for arm,p in [('B0',None),('B1',b1),('B2',b2),('B3',b3)]:
                errors=[]; rel=[]; thickness=[]; corrected=[]
                normalp=fit_plane(np.concatenate([C+raw[m,None]*rays(uvgrid[m],R) for (R,C),raw,m in zip(cams,allbase,masks)])) if p is None else p
                for i,((R,C),raw,m,zgt) in enumerate(zip(cams,allbase,masks,gt)):
                    cor=raw.copy()
                    if p is not None: cor[m]=depth(p,uvgrid[m],R,C)
                    delta=cor-raw
                    assert np.array_equal(cor[~m],raw[~m])
                    assert np.allclose(raw+delta,cor,rtol=0,atol=1e-12)
                    assert np.all(cor[m]>0)
                    pred,_=project(C+cor[m,None]*rays(uvgrid[m],R),R,C)
                    assert np.max(abs(pred-uvgrid[m]))<1e-8
                    errors.extend(abs(cor[m]-zgt[m])); rel.extend(abs(cor[m]-zgt[m])/zgt[m])
                    pts=C+cor[m,None]*rays(uvgrid[m],R)
                    own=fit_plane(pts); un=own/np.linalg.norm(own)
                    thickness.append(float(np.std(pts@un)))
                    f=folder/'arms'/arm; f.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(f/f'view{i}.npz',raw_depth=raw.reshape(H,W),residual_depth=delta.reshape(H,W),
                        corrected_depth=cor.reshape(H,W),mask=m.reshape(H,W))
                    corrected.append(cor)
                met={'plane':plane_id,'case':case,'arm':arm,'normal_deg':angle(normalp,truth),
                     'mae_m':float(np.mean(errors)),'relative_depth':float(np.mean(rel)),
                     'thickness_m':float(np.mean(thickness)),
                     'heldout_px':float(np.mean(np.linalg.norm(residual(normalp,hold,cams).reshape(-1,2),axis=1))),
                     'proposed_only':reason is not None,'retained':'B0' if reason else arm}
                if arm in ('B2','B3'): met['solver_success']=s2 if arm=='B2' else s3
                dump(folder/'arms'/arm/'metrics.json',met); rows.append(met)
                details[arm]=corrected[0]
            dump(folder/'observability.json',{'rejection':reason,'jacobian_singular_values':sv.tolist(),'analytic_vs_numeric_jacobian_relative_max':jac_error,'fit_count':len(fi),'holdout_count':len(vi)})
            checks[f'plane{plane_id}_{case}']=reason or 'SUPPORTED'
            if case=='tilt' and plane_id==1: example=(gt[0],allbase[0],details,masks[0])
    clean=[r for r in rows if r['arm']=='B3' and r['case'] in ('clean','tilt','offset','ripple')]
    passed=all(r['normal_deg']<.1 and r['relative_depth']<.001 and r['solver_success'] for r in clean)
    dump(out/'evaluation/metrics.json',rows)
    dump(out/'evaluation/checks.json',{'normal_recovery_passed':passed,'normal_cases':len(clean),
        'degeneracy':checks,'invariants':'positive depth, unchanged non-target, residual roundtrip, self projection passed'})
    truth,raw,det,mask=example
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    mid=(uvgrid[:,1]==119)&mask
    for name,z in [('GT',truth),('B0',raw)]+[(k,v) for k,v in det.items() if k!='B0']:
        ax[0].plot(uvgrid[mid,0],z[mid],label=name)
    ax[0].set(xlabel='u (pixel)',ylabel='Camera-Z (m)',title='Tilted wall: fixed image row'); ax[0].legend()
    for arm in ['B0','B1','B2','B3']:
        rr=[r for r in rows if r['plane']==1 and r['case']=='tilt' and r['arm']==arm][0]
        ax[1].bar(arm,rr['mae_m']*100)
    ax[1].set(ylabel='Depth MAE (cm)',title='Tilted wall, 12 degree raw normal error')
    fig.tight_layout(); fig.savefig(out/'figures/tilt_comparison.png',dpi=150); plt.close(fig)
    env={'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,
         'platform':platform.platform(),'execution':'sandbox CPU; no CUDA/GPU used','executable':sys.executable}
    dump(out/'environment.json',env)
    (out/'logs/summary.txt').write_text(f'normal_recovery_passed={passed}\n')
    files={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
    dump(out/'manifest.json',{'series':'planar_residual_depth','stage':'P0','run_id':rid,'status':'PASSED' if passed else 'FAILED',
         'parent_run':None,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'files':files})
    report=ROOT/'docs/experiments/planar_residual_depth/P0'/rid/'report.md'
    report.parent.mkdir(parents=True,exist_ok=True)
    lines=[f'# P0 합성 기하 검증 — {rid}', '',f'정상 조건 판정: **{"PASSED" if passed else "FAILED"}** ({len(clean)}개).',
           '', 'CPU 실행. VGGT-Ω·RGB 렌더링·SAM·실제 데이터 미사용. 단위 m, camera-Z. 정답 마스크·정확한 카메라·FIT 대응으로 보정하며 GT 평면/깊이는 평가에만 사용한다.',
           '', '| 평면 | 조건 | 군 | 법선 오차(°) | 깊이 MAE(cm) | 보류 재투영(px) |', '|---|---|---|---:|---:|---:|']
    for r in rows: lines.append(f"| {r['plane']} | {r['case']} | {r['arm']} | {r['normal_deg']:.6f} | {r['mae_m']*100:.6f} | {r['heldout_px']:.6f} |")
    lines+=['', '해석 Jacobian과 중앙 차분 비교의 상대 최대 오차 < 1e-6을 모든 조건에서 확인했다.', '', '## 범위와 제한','', '순수 회전·공선 대응은 관측/카메라 조건으로 감지해 retained=B0로 둔다. 해당 B2/B3 수치는 거절된 proposed 진단값이다. 오매칭은 FIT에만 20% 주입하며 보류 대응은 정확하다. Huber만 사용하고 RANSAC 전처리는 하지 않는다. 오매칭 조건은 정상 복원 합격 기준에 포함하지 않고 실패도 그대로 기록한다.',
            '', '평면은 z=3m 중심의 유한한 2×2m 사각형이며 세 방향(정면·경사벽·바닥형 경사면)을 사용한다. 실제 수평 바닥이나 카메라 위치 다양성을 모두 포괄하지 않는다. 자세 회전은 5°, baseline은 0.3m다. raw plane은 다중 시점 raw 점의 SVD 적합을 사용하며 ripple은 2.5cm 진폭이다. P0 무잡음 기하 검증용으로, 실데이터 robust fit 구현을 대체하지 않는다.',
            '', 'solver는 p=n/e의 3변수를 사용한다. B2는 p의 배율 1개만 갱신한다. radial Huber(delta=1px)를 IRLS와 least_squares로 계산한다. 시작점은 raw 적합 하나이며 실데이터 계획의 호모그래피 다중 초기값은 아직 구현하지 않았다. 법선은 부호 동치 각도로 평가한다.',
            '', f'결과: `results/planar_residual_depth/{rid}/`. 원본·잔차·보정 깊이·mask·입력 대응·카메라·GT·지표·소스·환경·hash를 실행별 저장했다. 원본 유지·잔차 복원·양의 깊이·자기 투영 검사는 assert로 수행했다.',
            '', '결론: 정상 합성 조건의 구현 검증이며 실제 VGGT 정확도 개선이나 이전 실험의 실패 원인 입증이 아니다.']
    report.write_text('\n'.join(lines)+'\n')
    idx=ROOT/'docs/experiments/planar_residual_depth/index.md'; idx.parent.mkdir(parents=True,exist_ok=True)
    with idx.open('a') as f: f.write(f'- P0 `{rid}`: {"PASSED" if passed else "FAILED"} — [보고서](P0/{rid}/report.md)\n')
    print(json.dumps({'run_id':rid,'passed':passed,'report':str(report),'figure':str(out/'figures/tilt_comparison.png')},ensure_ascii=False))
    if not passed: sys.exit(1)

if __name__=='__main__': main()
