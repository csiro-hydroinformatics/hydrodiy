# test function

ac1 <- function(innov, phi){
	n = length(innov)
	M = matrix(0, n, n)
	for(i in seq(1, n)) M[i, 1:i] = rev(phi^seq(0, i-1))

	return(M%*%innov)
}

runtest <- function(data, fbase){
    r = lm(y~x1+x2, data)

    fdata = sprintf("%s_data.csv", fbase)
    write.csv(data, fdata, quote=FALSE, row.names=FALSE)

    fres = sprintf("%s_result_estimate.csv", fbase)
    write.csv(summary(r)[[4]], fres, quote=FALSE, row.names=FALSE)

    rg = range(data$x1)
    x1 = seq(rg[1]-diff(rg), rg[2]+diff(rg), length=10)
    if(basename(fbase)=='linreg1'){
        rg = range(data$x2)
        x2 = seq(rg[1]-diff(rg), rg[2]+diff(rg), length=10)
        newdata = expand.grid(x1=x1, x2=x2)
    }else{
        newdata = data.frame(x1=x1, x2=x1^2, x3=x1^3)
    }

    pred = data.frame(newdata, predict(r, newdata, 
                                        interval='prediction'))

    fres = sprintf("%s_result_predict_gls.csv", fbase)
    write.csv(pred, fres, quote=FALSE, row.names=FALSE)
}


folder = "/home/magpie/Dropbox/code/pypackage/hydrodiy/hystat/tests"
folder = "D:\\code\\hydrodiy\\hydrodiy\\hystat\\tests"

# simple dataset
nval = 100
x1 = rnorm(nval)
x2 = rexp(nval)
y0 = 5+4*x1+3*x2
e = ac1(rnorm(nval)*0.4, 0.6)
y = y0+e
data = data.frame(x1, x2, y)
browser()
runtest(data, sprintf('%s/linreg1', folder))

e = ac1(rnorm(nval)*0.6, 0.95)
y = y0+e
data = data.frame(x1, x2, y)
runtest(data, sprintf('%s/linreg1', folder))

