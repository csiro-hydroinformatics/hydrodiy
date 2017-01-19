# test function
runtest <- function(data, fbase){
    res1 = acf(data, 5, plot=FALSE)
    res2 = acf(data, 5, plot=FALSE, type='covariance')
    
    fdata = sprintf("%s_data.csv", fbase)
    write.csv(data, fdata, quote=FALSE, row.names=FALSE)

    fres = sprintf("%s_result.csv", fbase)
    write.csv(data.frame(lag=res1$lag, acf=res1$acf, covar=res2$acf), 
              fres, quote=FALSE, row.names=FALSE)
}


# simple dataset
data = as.double(c(rep(1,5), rep(0,5), rep(1,5)))
runtest(data, 'data/acf1')

# acf dataset
data = lh
runtest(data, 'data/acf2')

data = arima.sim(n=200, list(ar=0.8), sd=1)
runtest(data, 'data/acf3')

data = arima.sim(n=500, list(ar=0.8), sd=1)
runtest(data, 'data/acf4')


