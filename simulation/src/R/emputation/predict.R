predict.emputation = function(object, Xtest, m=10){
  
  if (is.data.frame(Xtest)) Xtest = dftomat(Xtest)
  if (is.vector(Xtest) && is.numeric(Xtest)) Xtest <- matrix(Xtest, ncol = 1)
  
  if(object$standardize){
    Xtest  = sweep(sweep(Xtest,2,object$muX,FUN="-"),2,object$sddX,FUN="/")
  }
  
  Yhat1 = object$emputor(Xtest, m=m)  
  emp_list = lapply(1:m, function(t) Yhat1[t, , ])
  
  # de-standardize
  if (object$standardize) {
    emp_list = lapply(emp_list, function(mat) {
      sweep(sweep(mat, 2, object$sddX, "*"), 2, object$muX, "+")
    })
  }
  
  return(emp_list)
  
}
