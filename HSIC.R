tile<-function(G,i,j){
  row=rep(1:nrow(G),i)
  col=rep(1:ncol(G),j)
  return(G[row,col,drop=F])
}



rbf_dot<-function(pattern1, pattern2, deg){
  size1 = dim(pattern1)
  size2 = dim(pattern2)

  G = matrix(rowSums(pattern1 * pattern1),nrow = size1[1],ncol=1)
  H = matrix(rowSums(pattern2 * pattern2),nrow = size2[1],ncol=1)

  Q = tile(G, 1, size2[1])
  R = tile(t(H), size1[1], 1)

  H = Q + R - 2 * pattern1%*% t(pattern2)

  H = exp(-H / 2 / (deg ^ 2))

  return (H)
}

#' @title HSIC
#' @examples
#' X=matrix(c(0.1,0.1,1.0,2.0,3.0,4.0,5.0,2.0,3.0),ncol=1)
#' Y=matrix(c(0.2,0.2,1.0,1.0,2.0,2.0,3.0,2.0,3.0),ncol=1)
#' print(hsic_gam(X, Y))
#' @export
hsic_gam<-function(X, Y){
  n = dim(X)[1]

  # ----- width of X -----
  Xmed = X

  G = matrix(rowSums(Xmed ^2),n, 1)
  Q = tile(G, 1, n)
  R = tile(t(G), n, 1)

  dists = Q + R - 2 * (Xmed%*% t(Xmed))
  dists = dists[upper.tri(dists)]

  width_x = sqrt(0.5 * median(dists[dists > 0]))
  # ----- -----

  # ----- width of X -----
  Ymed = Y

  G = matrix(rowSums(Ymed ^2),n, 1)
  Q = tile(G, 1, n)
  R = tile(t(G), n, 1)

  dists = Q + R - 2 * (Ymed%*% t(Ymed))
  dists = dists[upper.tri(dists)]

  width_y = sqrt(0.5 * median(dists[dists > 0]))
  # ----- -----

  bone = matrix(1,n,1)
  H = diag(1,n) - matrix(1,n,n) / n

  K = rbf_dot(X, X, width_x)
  L = rbf_dot(Y, Y, width_y)

  Kc = H%*% K%*% H
  Lc = H%*% L%*% H

  testStat = sum(t(Kc) * Lc) / n

  varHSIC = (Kc * Lc / 6) ^ 2

  varHSIC = (sum(varHSIC) - sum(diag(varHSIC))) / n / (n - 1)

  varHSIC = varHSIC * 72 * (n - 4) * (n - 5) / n / (n - 1) / (n - 2) / (n - 3)

  K = K - diag(diag(K))
  L = L - diag(diag(L))

  muX = t(bone)%*% K%*% bone / n / (n - 1)
  muY = t(bone)%*% L%*% bone / n / (n - 1)

  mHSIC = (1 + muX * muY - muX - muY) / n

  al = mHSIC ^ 2 / varHSIC
  bet = varHSIC * n / mHSIC

  pval <- 1 - pgamma(q = testStat, shape = al, rate = 1/bet)

  #alph=pval
  #thresh = gamma.ppf(1 - alph, al, scale=bet)[0][0]

  return (list(testStat=testStat, pval=pval))
  }


