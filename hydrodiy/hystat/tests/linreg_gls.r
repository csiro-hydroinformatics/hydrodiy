library(orcutt)


# Generate AR1 series from innovations
ac1 <- function(nval, phi, scale){
	
	# Generate innovation
	innov = rnorm(nval) * scale

	# Generate AR1
	M = matrix(0, nval, nval)
	for(i in seq(1, nval)) M[i, 1:i] = rev(phi^seq(0, i-1))

	# Center
	innov = innov - mean(innov)

	return(M%*%innov)
}

# test function
runtest <- function(data, fbase){

    r = lm(y~x1+x2, data)
    sr = cochrane.orcutt(r)

    fdata = sprintf("%s_gls_data.csv", fbase)
    write.csv(data, fdata, quote=FALSE, row.names=FALSE)

    fres = sprintf("%s_gls_result_estimate_ols.csv", fbase)
    params_ols = summary(r)$coefficients
    write.csv(params_ols, fres, quote=FALSE, row.names=FALSE)

    fres = sprintf("%s_gls_result_estimate_gls.csv", fbase)
    params_gls = sr$Cochrane.Orcutt$coefficients
    write.csv(params_gls, fres, quote=FALSE, row.names=FALSE)

    rg = range(data$x1)
    x1 = seq(rg[1]-diff(rg), rg[2]+diff(rg), length=10)
    rg = range(data$x2)
    x2 = seq(rg[1]-diff(rg), rg[2]+diff(rg), length=10)
    newdata = expand.grid(x1=x1, x2=x2)

    pred = newdata 
    M = cbind(matrix(1, dim(pred)[1], 1), data.matrix(newdata))
    pred$ols = M %*% data.matrix(params_ols[,1])
    pred$gls = M %*% data.matrix(params_gls[,1])

    fres = sprintf("%s_gls_result_predict_gls.csv", fbase)
    write.csv(pred, fres, quote=FALSE, row.names=FALSE)
}

# Folder
folder = "/home/magpie/Code/pypackage/hydrodiy/hystat/tests"
if(!file.exists(folder)) folder = "D:\\code\\hydrodiy\\hydrodiy\\hystat\\tests"

# simple dataset
nval = 100
x1 = rnorm(nval)
x2 = rexp(nval)*0.5
y0 = 5+4*x1+3*x2
e = ac1(nval, 0.7, 3)
y = y0+e
data = data.frame(x1, x2, y)
runtest(data, sprintf('%s/linreg1', folder))

e = ac1(nval, 0.9, 5)
y = y0+e
data = data.frame(x1, x2, y)
runtest(data, sprintf('%s/linreg1', folder))

