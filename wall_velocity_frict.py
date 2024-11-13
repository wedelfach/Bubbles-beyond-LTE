# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 13:34:27 2024

@author: User
"""



import numpy as np
from scipy.integrate import solve_ivp , simps
from scipy.optimize import brentq


def solve(g,arg={},a=0,b=1,N=100,acc=0.001):

    """
    Finds the smallest solution of an equation g(x)=0 such that a < x_sol < b.

    Parameters
    ----------
    g : callable
        The scalar function `g(x)` encapsulates LHS of the equation.
    arg : iterable, optional
        The parameters passed to function g
    a : float, optional
        The lower limit of the search region.
    b : float, optional
        The upper limit of the search region.
    N : integer, optional
        Numer of samples on the [a, b] set that controls the resolution.
    acc : float, optional
        The maximum allowed value is taken by g(x) at the 
        solution found by the solver. If not reached the accuracy error 
        is raised.

    Returns
    -------
      sol: float
        The solution found by the routine

    Notes
    -----
      The algorithm is a basic implementation of the sign-change
      routine. Once the two points between whom the sign of g(x) changes
      are identified scipy brentq is used to compute the solution accurately.
      N=100 is usually sufficient for Bag Model were c_s^2=c_b^2=1/3. 
      For different sound velocities, one may consider choosing a higher N.

    """
    def f(u):
        try:
            return g(u,*arg)
        except ValueError:
            return None
        
    #Check which probing of the interval [a, b] is more suitable
    if a <= 0 or b <= 0 or np.abs(np.log10(a/b)) < 1:
        x=np.linspace(a,b,N)
    else:
        x=np.logspace(np.log(a),np.log(b),N,base=np.e)
    
    #Compute the function g(x) at N points
    l1=np.array(list(map(f,x)))
    x=x[l1 != None]
    l1=l1[l1 != None]
    
    #Use sign change rutine
    mem=l1[0]
    sol=None
    for _it in range(len(l1)):
        #Sign change found! Use brentq to find the solution.
        if np.sign(mem)*np.sign(l1[_it]) == -1:
            if mem > l1[_it]:
                sol = brentq(f,  a=x[_it], b=x[_it-1])
                assert np.abs(f(sol)) < acc
                break
            else:
                sol = brentq(f, a=x[_it-1], b=x[_it])
                assert np.abs(f(sol)) < acc
                break
        mem=l1[_it]
    if sol==None:
        raise ValueError #No solution found, raise ValueError.
    return sol

"""
Functions computing transition parameters. For documentation 
check Appendix C of arXiv:2303.10171. Two modifications were made:
    1) The original solving algorithm was replaced by the custom one.
    2) The third matching condition was implemented in the function "eqWall" 
    and "detonation"
    3) Function "detonation" always returns the physical solution. In case 
    two steady states are found, detonation chooses a faster stable solution
    and ignores the unstable slower one. In the LTE limit, no stable detonation
    exists (the walls run away) and thus detonation always returns None.
    4) In the nu-mu model with non-zero friction we often find two solutions 
    for alpha_+. We verified that it is the bigger one that leads to the 
    solutions consistent with hydrodynamic simulations.
    5) In "eqWall" we always choose the smaller solution for v_p,
    which corresponds to the (-1) branch.
"""

def find_vJ (alN ,cb2 ):
 return np. sqrt (cb2 ) *(1+ np. sqrt (3* alN *(1 - cb2 +3* cb2*alN))) /(1+3* cb2*alN)

def get_vp (vm ,al ,cb2 , branch = -1):
 disc = vm **4 -2* cb2 *vm **2*(1 -6* al)+cb2 **2*(1 -12* vm **2* al *(1 -3* al))
 return 0.5*( cb2+vm **2+ branch *np. sqrt ( disc ))/( vm +3* cb2*vm*al)

def w_from_alpha (al ,alN ,nu ,mu):
  return (abs((1 -3* alN)*mu -nu)+1e-100) /(abs((1 -3* al)*mu-nu)+1e-100)

def eqWall (al ,alN ,vm ,nu ,mu ,psiN, vw, rho, solution = -1):
     vp=get_vp (vm ,al ,1/( nu -1) , -1)
     ga2m , ga2p = 1/(1 - vm **2) ,1/(1 - vp **2)
     psi = psiN * w_from_alpha (al ,alN ,nu ,mu)**( nu/mu -1)
     #Third matching condition: (T_m/T_p)=r at refraction front
     r=ga2p/ga2m*(1-rho*abs(ga2p)**.5*vp/(w_from_alpha (al ,alN ,nu ,mu)+1e-100))**2  
     return vp*vm*al /(1 -(nu -1) *vp*vm+1e-100) -(1 -3*al -( r )**( nu/2)*psi) /(3* nu)

def solve_alpha (vw ,alN ,cb2 ,cs2 , psiN, rho ):
     nu ,mu = 1+1/ cb2 ,1+1/ cs2
     vm = min(np. sqrt (cb2),vw)
     vp_max = min (cs2/vw ,vw)
     al_min = max ((vm - vp_max )*( cb2 -vm* vp_max ) /(3* cb2*vm *(1 - vp_max **2) ) ,(mu -nu)/(3* mu))
     al_max = 1/3
     branch = -1
     sol1 = solve (eqWall ,( alN ,vm ,nu ,mu ,psiN, vw, rho, branch ), al_min , al_max)
     try:
        sol=solve (eqWall ,( alN ,vm ,nu ,mu ,psiN, vw, rho, branch ), sol1*(1+1e-5), al_max)
     except ValueError or AssertionError:
        sol=sol1
     return sol

def dfdv (v,X,cs2 ):
     xi ,w = X
     mu_xiv = (xi -v)/(1 - xi*v)
     dxidv = xi *(1 -v*xi)*( mu_xiv **2/ cs2 -1) /(2* v*(1 -v **2)+1e-100 )
     dwdv = w *(1+1/ cs2 )* mu_xiv /(1 -v **2)
     return [dxidv , dwdv ]

def integrate_plasma (v0 ,vw ,w0 ,c2 , shock_wave = True ):
     def event (v,X,cs2):
         xi ,w = X
         return xi *(xi -v)/(1 - xi*v) - cs2
     event.terminal = True
     sol = None
     if shock_wave :
         sol = solve_ivp (dfdv ,(v0 ,1e-20) ,[vw ,w0], events =event , args =(c2 ,) ,rtol =1e-10 , atol =1e-10)
     else :
         sol = solve_ivp (dfdv ,(v0 ,1e-20) ,[vw ,w0], args =(c2 ,) ,rtol =1e-10 , atol =1e-10)
     if not sol.success :
         print (" WARNING : desired precision not reached in ’ integrate_plasma ’")
     return sol

def shooting (vw ,alN ,cb2 ,cs2 , psiN, rho):
     nu ,mu = 1+1/ cb2 ,1+1/ cs2
     vm = min(np. sqrt (cb2),vw)
     al = solve_alpha (vw , alN , cb2 , cs2 , psiN, rho )         
     vp = get_vp (vm , al , cb2 )
     wp = w_from_alpha (al , alN , nu , mu)
     sol = integrate_plasma ((vw -vp)/(1 - vw*vp), vw , wp , cs2)
     vp_sw = sol.y[0 , -1]
     vm_sw = (vp_sw -sol.t[ -1]) /(1 - vp_sw * sol.t[ -1])
     wm_sw = sol.y[1 , -1]
     return vp_sw / vm_sw - ((mu -1)* wm_sw +1) /((mu -1)+ wm_sw )

def find_vw (alN ,cb2 ,cs2 , psiN, rho=0, vw_max=None):
     nu ,mu = 1+1/ cb2 ,1+1/ cs2
     vJ = find_vJ (alN , cb2 )
     if alN < (1- psiN )/3 or alN <= (mu -nu) /(3* mu):
         print ("alN too small")
         return None
     if alN > max_al (cb2 ,cs2 ,psiN ,100, rho) or shooting (vJ ,alN ,cb2 ,cs2 , psiN, rho ) < 0:
         print ("alN too large, no stable deflagration/hybrid solutions")
         return None
     if vw_max==None:
         sol = solve( shooting ,( alN ,cb2 ,cs2 , psiN, rho ),a=0.001, b=vJ)
     else:
         sol = solve( shooting ,( alN ,cb2 ,cs2 , psiN, rho ),a=0.001, b=vw_max)
     return sol

def max_al (cb2 ,cs2 ,psiN , upper_limit =1, rho=0) :
     nu ,mu = 1+1/ cb2 ,1+1/ cs2
     vm = np. sqrt (cb2 )
     def func (alN, rho):
         vw = find_vJ (alN , cb2 )
         vp = cs2 /vw
         ga2p , ga2m = 1/(1 - vp **2) ,1/(1 - vm **2)
         wp = (vp+vw -vw*mu)/( vp+vw -vp*mu)
         psi = psiN *wp **( nu/mu -1)
         al = (mu -nu) /(3* mu)+( alN -(mu -nu) /(3* mu))/wp
         r=ga2p/ga2m*(1-rho*abs(ga2p)**.5*vp/(w_from_alpha (al ,alN ,nu ,mu)+1e-100))**2
         return vp*vm*al /(1 -(nu -1) *vp*vm) -(1 -3*al -( r )**( nu /2)*psi) /(3*nu)
     if func ( upper_limit, rho ) < 0:
         return upper_limit
     sol = solve( func ,[rho],a=(1 - psiN )/3, b=upper_limit)
     return sol

def detonation (alN ,cb2 , psiN, rho ):
     nu = 1+1/ cb2
     vJ = find_vJ (alN , cb2 )
     def matching_eq (vw):
         A = vw **2+ cb2 *(1 -3* alN *(1 - vw **2) )
         vm = (A+np. sqrt (A**2 -4* vw **2* cb2 )) /(2* vw)
         ga2w , ga2m = 1/(1 - vw **2) ,1/(1 - vm **2)
         r=(ga2w / ga2m)*1/(1+rho*vw*ga2w**.5)**2
         return vw*vm*alN /(1 -(nu -1) *vw*vm) -(1 -3* alN -( r )**( nu /2)* psiN )/(3* nu)

     sol_det_1 = solve( matching_eq, arg=(),a= vJ +1e-10,b=1 -1e-10)
     if matching_eq (sol_det_1-1e-5)>0 and matching_eq (sol_det_1+1e-5)<0:
         return sol_det_1
     try:
         sol_det_2 = solve( matching_eq, arg=(),a= max(sol_det_1 +1e-10, vJ) ,b=1-1e-10)
         return sol_det_2
     except ValueError:
         print("No stable detonations found!")
         return None

def find_kappa (alN ,cb2 ,cs2 ,psiN ,vw= None ):
     if vw is None :
         vw = find_vw (alN ,cb2 ,cs2 , psiN )
         nu ,mu = 1+1/ cb2 ,1+1/ cs2
         kappa ,wp ,vm ,vp = 0 ,1 ,0 ,0
     if vw < 1:
         vm = min (np. sqrt (cb2),vw)
         al = solve_alpha (vw ,alN ,cb2 ,cs2 , psiN )
         vp = get_vp (vm ,al ,cb2 )
         wp = w_from_alpha (al ,alN ,nu ,mu)
         sol = integrate_plasma ((vw -vp)/(1 - vw*vp),vw ,wp ,cs2)
         v,xi ,w = sol .t,sol .y[0] , sol .y[1]
         kappa += 4* simps (( xi*v) **2* w/(1 -v **2) ,xi)/( vw **3* alN)
     if vw **2 > cb2 :
         w0 = psiN *wp **( nu/mu)*((1 - vm **2) /(1 - vp **2) )**( nu /2) if vw < 1 else 1+6* alN /(nu -2)
         v0 = (vw -vm)/(1 - vw*vm) if vw < 1 else 3* alN /(nu -2+3* alN )
         sol = integrate_plasma (v0 ,vw ,w0 ,cb2 , False )
         v,xi ,w = np. flip (sol .t),np. flip (sol.y [0]) ,np. flip (sol.y [1])
         mask = np. append (xi [1:] > xi [: -1] , True )
         kappa += 4* simps ((( xi*v) **2* w/(1 -v **2) )[ mask ],xi[ mask ]) /( vw **3* alN )
     return kappa

