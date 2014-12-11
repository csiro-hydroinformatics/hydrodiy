# test function
runtest <- function(data, fbase){
    if(basename(fbase)=='linreg1'){
        r = lm(y~x1+x2, data)
    }
    else{
        r = lm(y~x1+x2+x3, data)
    }

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

    fres = sprintf("%s_result_predict.csv", fbase)
    write.csv(pred, fres, quote=FALSE, row.names=FALSE)
}

# Folder
folder = "/home/magpie/Code/pypackage/hydrodiy/hystat/tests"
if(!file.exists(folder)) folder = "D:\\code\\hydrodiy\\hydrodiy\\hystat\\tests"

# simple dataset
nval = 100
x1 = rnorm(nval)
x2 = rexp(nval)
y0 = 5+4*x1+3*x2
y = y0+5*rnorm(nval)
data = data.frame(x1, x2, y)
runtest(data, sprintf('%s/linreg1', folder))

y0 = 50+10*x1+5*x1^2+4*x1^3
y = y0+40*rnorm(nval)
data = data.frame(x1=x1, x2=x1^2, x3=x1^3, y)
runtest(data, sprintf('%s/linreg2', folder))

