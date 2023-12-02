### A Pluto.jl notebook ###
# v0.19.29

using Markdown
using InteractiveUtils

# ╔═╡ a8577720-7990-11ee-3a5e-eb35a337d22d
begin
	using KernelDensity
	using Plots
	using PlutoUI
	using LinearAlgebra
	using Distributions
	using DistributionsAD
	using Todo
	using ForwardDiff
	using HCubature
end

# ╔═╡ 75503983-557f-4926-9f2e-478d381bf731
md"""# SVGD 
This is a short implementation of basic **SVGD** principles for a standard rosenbrock distribution. The functions as well as the parameters can be tweaked/replaced as necessary for the task at hand.(Gaussian/Rastrigin also provided)
"""

# ╔═╡ e36d0610-c7b7-4cb4-86f4-485b7dd9bd7b
np=6

# ╔═╡ 408009b6-3f3c-4cb9-a4d0-8ef4da948b84
ni=1000

# ╔═╡ 44c16b2d-5e33-4f99-972e-0506065fca63
stepsize=.1

# ╔═╡ 2d45645a-2798-42fa-9399-a110cd890e70
md"""## Functions
First of all the different functions which will be used need to be defined."""

# ╔═╡ acb67c33-0b23-459e-aa82-a84687a15de2
begin
function gauss(xj)
	n=2
	σ=2
	μ=zeros(n,1)
	return 1/(σ*sqrt((2*pi)^n))*exp(-norm(xj-μ)^2/(2*σ^2))
end
end

# ╔═╡ 8c999fa4-b887-44d1-831f-0220c3759cda
begin
function loggauss(xj)
	f=log(gauss(xj))
	return f
end
end

# ╔═╡ 597721db-82fe-40b0-bd79-695f7d1f1a27
begin
gradloggauss(x::T) where{T<:Real} = ForwardDiff.derivative(x->loggauss(x),x)
gradloggauss(x::AbstractArray{T}) where{T<:Real} = ForwardDiff.gradient(x->loggauss(x),x)
end

# ╔═╡ 27b5dc62-c39e-4997-9ea3-47e6831799aa
function rastrigin(x)
	A=10.0
	n=length(x)
	f=A*n
	for xi in x
		f+= xi^2 - A*(cos(2*pi*xi))
	end
	return f
end

# ╔═╡ 4a97a49e-7b9d-4563-b540-3a65e001a1c7
lograst(x) = log(rastrigin(x))

# ╔═╡ 15078951-4950-42c0-ae07-c6092fc8ada9
gradlograst = x -> ForwardDiff.gradient(lograst,x)

# ╔═╡ 0ca64d1c-a648-4e79-ac77-1861604e73cb
rosenbrock(x) = sqrt(0.05*5)/π*exp(-((1.0 - x[1])^2 + 100 * (x[2] - x[1]^2)^2)/20)

# ╔═╡ 95536a80-43dd-4fda-9e97-a1b71ccfea57
logros(x) = log(rosenbrock(x))

# ╔═╡ f2f14f9f-8bfd-4642-ac69-1db28298d8ab
gradlogros(x) = ForwardDiff.gradient(x->logros(x),x)

# ╔═╡ 331f6880-27d3-4d42-9039-98a2066a8fdf
begin
function k(x1,x2)
	h=1
	return exp(-norm(x1-x2)^2/h)
end
end

# ╔═╡ 3f5cbd17-2c3b-4149-8422-bfb6845177a0
begin
gradk(x1::T,x2) where {T<:Real}= ForwardDiff.derivative(x1->k(x1,x2),x1)
gradk(x1::AbstractArray{T},x2) where {T<:Real} = ForwardDiff.gradient(x1->k(x1,x2),x1)
end

# ╔═╡ f4f8e0a8-3fb9-4350-8b56-7507fb82ca6e
begin
function ϕ(xi,x)
	n=2
	sum=zeros(2,1)
	for j in 1:np^2
		sum+=k(x[j,:],xi)*gradlogros(x[j,:])+gradk(x[j,:],xi) 
	end
	return 1/size(x,1)*sum
end
end

# ╔═╡ 273f06ed-4ea0-499a-9cbc-9728ff5a6d38
md"""### SVGD
Iterative method for approximation proposed in (Liu, Wang 2016)."""

# ╔═╡ 17fecc47-6328-4e40-8cd7-08aa15aa6e1e
md"""##### Grid methods"""

# ╔═╡ c6a5665e-2b18-49e9-93eb-c44a8afff73b
function ndgrid()
	tuples = Iterators.product(range(-1,1,np), range(-1,1,np))
	tuples = vec(collect(tuples))
	grid=zeros(length(tuples),2);
	for i in 1:length(tuples)
		grid[i,:]=[j for j in tuples[i]]
	end
	return grid
end

# ╔═╡ a6219cca-c24a-40f5-9e17-749f115050a2
function meshgrid(x, y)
    aa = x' .* ones(length(x))
    bb = ones(length(y))' .* y
    return aa, bb
end

# ╔═╡ 394b9dde-fd2c-423f-8502-5b7ad233795b
md"""##### Initialisation"""

# ╔═╡ 8d078e0f-862a-42e0-8475-565ad512b14c
begin
ϵ= stepsize
particles=ndgrid()
temp=zeros(np^2,2)
Iterations=zeros(ni,np^2,2)
end;

# ╔═╡ 11256e61-56fa-4169-bdbc-750201e14fe0
md"""##### Iterative update process"""

# ╔═╡ 81d75374-db13-465d-9fd5-ca5efab645e2
begin
for j in 1:ni
	for i in 1:np^2
		temp[i,:]=particles[i,:]+ϵ*ϕ(particles[i,:],particles)
	end	
	particles=copy(temp)
	Iterations[j,:,:]=copy(temp)
end
end

# ╔═╡ b525d418-5ea7-460d-ad6a-39866ff9a828
md"""##### Expected values and variance for each iteration"""

# ╔═╡ 78d7f3c8-cd1b-45f1-8260-4011949f4cc1
begin
expected=zeros(ni,2)
variance=zeros(ni,2)
covariance=zeros(ni,1)
for j in 1:ni
	for i in 1:np^2
		expected[j,:]+=Iterations[j,i,:]
	end
	expected[j,:]=expected[j,:]/(np^2-1)
	for i in 1:np^2
		variance[j,:]+=(Iterations[j,i,:]-expected[j,:]).^2
		covariance[j]+=(Iterations[j,i,1]-expected[j,1])*(Iterations[j,i,2]-expected[j,2])
	end
	covariance[j]=covariance[j]/(np^2-1)
	variance[j,:]=variance[j,:]/(np^2-1)
end
end

# ╔═╡ fb59e3b3-8ace-45d3-9a78-5ff2213ad810
md"""##### Vizualization of marginaization"""

# ╔═╡ e89a305d-6ce2-4fbd-bdb3-7ea79a1b57ff
function marginalization(f,n)
	fineness=0.2
	points=200
	margs=zeros(points,n)
	if n==2
			for i in 1:points
				xval=-8+fineness*(i-1)
				yval=-3+0.5fineness*(i-1)
				rintx(x)=f([xval,x])
				rinty(x)=f([x,yval])
				margs[i,1]=hquadrature(rintx,-100,100)[1]
				margs[i,2]=hquadrature(rinty,-5,10)[1]
			end
	else
		for j in 0:n-1
			for i in 1:points
				ü=-3+fineness*(i-1)
				rint(x)=f(vcat(x[1:j],ü,x[j+1:n-1]))
				margs[i,j+1]=hcubature(rint,[-5 for i in 1:n],[5 for i in 1:n])[1]
			end
		end
	end
	return margs
end

# ╔═╡ 060b4a6c-6a0d-4024-a2d1-26c2d3f256b2


# ╔═╡ e2807f19-900f-47bc-98a5-4c933152b28d
md"""Test of kde package"""

# ╔═╡ 096637c3-746f-4c37-bdcb-556f0bd55bb3
pckest=kde(Iterations[ni,:,:])

# ╔═╡ bc615b09-de96-4d02-bc4e-2037026e3297
begin
aa = pckest.x
bb = pckest.y
a_grid = repeat(reshape(aa, 1, :), length(bb), 1)
b_grid = repeat(bb, 1, length(aa))
end;

# ╔═╡ 99aacbad-8f14-4f58-80b1-15c978689d65
contkde=contour(aa,bb,pckest.density')

# ╔═╡ e10f83fb-adc0-416a-80c2-b818520f435f
md"""Own KDE method"""

# ╔═╡ 403657c1-b3c8-48f9-bc27-bc4c12bd7321
function KDE(x)
	n=size(temp,1)
	dims=2
	f_h=0
	h=1
	for i in 1:n
		f_h+=1/(h*sqrt((2*pi)^dims))*exp(-norm(x-temp[i,:])/h)
	end
	return f_h/n
end

# ╔═╡ 3f0c058d-d355-4a8b-a633-e900afcd1fa1
begin
	margs=marginalization(rosenbrock,2);
	margKDE=marginalization(KDE,2)
	xvec=[-8+0.2i for i in 0:100]
end;

# ╔═╡ fee09141-0f47-42b5-959d-9641436512fc
begin
	scatter(xvec,margKDE[:,1],label="KDE")
	scatter!(xvec,margs[:,1],label="rosenbrock")
end;

# ╔═╡ 50c19afd-3194-4a84-83b6-f617f272d0a8
begin
	cdf=zeros(100)
	cdfKDE=zeros(100)
	diff=zeros(100)
	for i in 2:100
		cdfKDE[i]=cdfKDE[i-1]+margKDE[i,1]*0.2
		cdf[i]=cdf[i-1]+margs[i,1]*0.2	
	end	
end

# ╔═╡ ffcfdda4-b163-404b-8d17-22fa86090674
begin
	sorted=sort(temp[:,1])
	vals=[1/np^2*i for i in 1:np^2]
	scatter(sorted,vals,label="SVGD")
	scatter!(xvec,cdf,label="cdf: rosenbrock")
	scatter!(xvec,cdfKDE,label="cdf: KDE")
end;

# ╔═╡ 2b7f02b2-6029-44cf-85b5-7759e8a60279
md"""##### Visualization of SVGD
from https://docs.juliaplots.org/latest/gallery/gr/generated/gr-ref022/#gr_ref022"""

# ╔═╡ fccf6474-aab9-471b-bf69-73ac24d306d6
begin
a = -10:0.05:10
b = -1:0.05:100
f(x, y) = begin
        rosenbrock([x,y])#p([x,y])
    end
x_grid = repeat(reshape(a, 1, :), length(b), 1)
y_grid = repeat(b, 1, length(a))
mapping_p = map(f, x_grid, y_grid)
cont1 = contour(a, b, mapping_p, fill = true)
cont2 = contour(a, b, mapping_p,fill =(true,cgrad(:oxy, scale=:log)))
end;

# ╔═╡ 9b52a48c-30ba-4903-8728-240917a95005
md"""#### Comparison to KDE/true density function
(blue dot represents expected value of SVGD approximation)"""

# ╔═╡ f941eb31-d2b4-47e2-a6a5-7d5bcd8e19a7
# ╠═╡ disabled = true
#=╠═╡
animation=@animate for i in 1:ni
	plot(cont2)
	scatter!(Iterations[i,:,1],Iterations[i,:,2],legend=false,color="orange")
	scatter!((expected[i,1],expected[i,2]),legend=false,color="blue")
	xlims!(-3,3)
	ylims!(-1,3)
end
  ╠═╡ =#

# ╔═╡ 46ef453d-6538-46bf-8522-3d61a2807701
#=╠═╡
gif(animation,fps=20)
  ╠═╡ =#

# ╔═╡ f2103fd9-8f3c-48ca-b36c-1e95b5b43e09
begin
plot(contkde)
scatter!((Iterations[ni,:,1],Iterations[ni,:,2]))
scatter!((expected[ni,1],expected[ni,2]),legend=false,color="blue")
end

# ╔═╡ fd2bc0f5-7d55-4620-a128-e22fc69c6d66
md"""###### Expected values/variances at different iterations"""

# ╔═╡ 8441bb4d-8c9f-459d-9241-94378f1045b2
begin
	scatter(expected[:,1],expected[:,2],label="expected values",color="dark red")
end

# ╔═╡ 54a48193-12e9-48e2-8d33-c9f769d114c0
begin	
	scatter(variance[:,1],label="var(X)",color="blue")
	scatter!(variance[:,2],label="var(Y)",color="green")
	scatter!(covariance[:],label="cov(X,Y)",color="red")
end

# ╔═╡ Cell order:
# ╟─75503983-557f-4926-9f2e-478d381bf731
# ╠═a8577720-7990-11ee-3a5e-eb35a337d22d
# ╠═e36d0610-c7b7-4cb4-86f4-485b7dd9bd7b
# ╠═408009b6-3f3c-4cb9-a4d0-8ef4da948b84
# ╠═44c16b2d-5e33-4f99-972e-0506065fca63
# ╟─2d45645a-2798-42fa-9399-a110cd890e70
# ╟─acb67c33-0b23-459e-aa82-a84687a15de2
# ╟─8c999fa4-b887-44d1-831f-0220c3759cda
# ╟─597721db-82fe-40b0-bd79-695f7d1f1a27
# ╟─27b5dc62-c39e-4997-9ea3-47e6831799aa
# ╟─4a97a49e-7b9d-4563-b540-3a65e001a1c7
# ╟─15078951-4950-42c0-ae07-c6092fc8ada9
# ╟─0ca64d1c-a648-4e79-ac77-1861604e73cb
# ╟─95536a80-43dd-4fda-9e97-a1b71ccfea57
# ╟─f2f14f9f-8bfd-4642-ac69-1db28298d8ab
# ╟─331f6880-27d3-4d42-9039-98a2066a8fdf
# ╟─3f5cbd17-2c3b-4149-8422-bfb6845177a0
# ╟─f4f8e0a8-3fb9-4350-8b56-7507fb82ca6e
# ╟─273f06ed-4ea0-499a-9cbc-9728ff5a6d38
# ╟─17fecc47-6328-4e40-8cd7-08aa15aa6e1e
# ╟─c6a5665e-2b18-49e9-93eb-c44a8afff73b
# ╟─a6219cca-c24a-40f5-9e17-749f115050a2
# ╟─394b9dde-fd2c-423f-8502-5b7ad233795b
# ╟─8d078e0f-862a-42e0-8475-565ad512b14c
# ╟─11256e61-56fa-4169-bdbc-750201e14fe0
# ╟─81d75374-db13-465d-9fd5-ca5efab645e2
# ╟─b525d418-5ea7-460d-ad6a-39866ff9a828
# ╟─78d7f3c8-cd1b-45f1-8260-4011949f4cc1
# ╟─fb59e3b3-8ace-45d3-9a78-5ff2213ad810
# ╟─e89a305d-6ce2-4fbd-bdb3-7ea79a1b57ff
# ╟─3f0c058d-d355-4a8b-a633-e900afcd1fa1
# ╟─fee09141-0f47-42b5-959d-9641436512fc
# ╠═060b4a6c-6a0d-4024-a2d1-26c2d3f256b2
# ╠═50c19afd-3194-4a84-83b6-f617f272d0a8
# ╟─ffcfdda4-b163-404b-8d17-22fa86090674
# ╟─e2807f19-900f-47bc-98a5-4c933152b28d
# ╠═096637c3-746f-4c37-bdcb-556f0bd55bb3
# ╟─bc615b09-de96-4d02-bc4e-2037026e3297
# ╠═99aacbad-8f14-4f58-80b1-15c978689d65
# ╟─e10f83fb-adc0-416a-80c2-b818520f435f
# ╟─403657c1-b3c8-48f9-bc27-bc4c12bd7321
# ╟─2b7f02b2-6029-44cf-85b5-7759e8a60279
# ╠═fccf6474-aab9-471b-bf69-73ac24d306d6
# ╟─9b52a48c-30ba-4903-8728-240917a95005
# ╟─f941eb31-d2b4-47e2-a6a5-7d5bcd8e19a7
# ╟─46ef453d-6538-46bf-8522-3d61a2807701
# ╠═f2103fd9-8f3c-48ca-b36c-1e95b5b43e09
# ╟─fd2bc0f5-7d55-4620-a128-e22fc69c6d66
# ╠═8441bb4d-8c9f-459d-9241-94378f1045b2
# ╠═54a48193-12e9-48e2-8d33-c9f769d114c0
