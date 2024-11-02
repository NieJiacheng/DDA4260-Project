# RBM

## Naive RBM

###basic assumption

$$
\begin{flalign*}

&
\mathcal{v} \in \{0, 1\}^{d_{v} \times d_{k}}, d_{v} \ \text{is the number of rating taregt,} d_{k} \ \text{ is the scale of rating}
&

\end{flalign*}
$$

$$
\begin{flalign*}

&

\mathcal{w} \in \mathcal{R}^{d_{h} \times d_{v} \times d_{k}}, d_{h}\ \text{is the number of hidden state}

&
\end{flalign*}
$$

$$
\begin{flalign*}
&
\mathcal{h} \in \{0, 1\}^{d_{h}}
&
\end{flalign*}
$$

--------



$$
\begin{flalign*}

&
\text{For binary } \mathcal{v} {,} \mathcal{h} \text{, define energy function:} \\

&
E(\mathcal{v}, \mathcal{h};\mathcal{w})

=  -\sum_{i, j, k} \mathcal{w}_{ijk}h_{i}v_{jk} 

= \mathcal{h}^{T}\mathcal{w}\mathcal{v} \\

&
\text{where } \mathcal{w}\mathcal{v}  \text{ is }  einsum(\mathcal{w}, \mathcal{v}, ijk, jk \rightarrow i)
&

\end{flalign*}
$$



$$
\mathbf{P}(\mathcal{v}, \mathcal{h}; \mathcal{w}) = 
\frac{e^{-E(\mathcal{v}, \mathcal{h}; \mathcal{w})}}{Z}

\text{ where Z is the marginal } \sum_{\mathcal{v}, \mathcal{h}}e^{-E(\mathcal{v}, \mathcal{h}; \mathcal{w})}
$$



$$
\begin{flalign*}
& 
\text{Note that}\ \mathcal{h_{i}} \ \text{are supposed to be independent for given $v$}.
&
\\
&
\Rightarrow \mathbf{P}(\mathcal{h}| \mathcal{v}; \mathcal{w}) = \prod_{i}\mathbf{P}(h_{i}| v; \mathcal{w}) 
\\
\\
&

\text{def } E(v, h_{i}; w) = -\sum_{j, k} \mathcal{w}_{ijk}h_{i}v_{jk} \\

&

\text{i.e. } \sum_{i} E(v, h_{i}; w) = E(v, h  ; w)

\\



&

\mathbf{P}(\mathcal{h}| \mathcal{v}; \mathcal{w}) 
= \frac{\mathbf{P}(\mathcal{h}, \mathcal{v}; \mathcal{w})}{\sum_{h} \mathbf{P}(\mathcal{h}, \mathcal{v}; \mathcal{w}) }
= \frac{\frac{e^{-E(\mathcal{v}, \mathcal{h}; \mathcal{w})}}{Z}}{\sum_{h}\frac{e^{-E(\mathcal{v}, \mathcal{h}; \mathcal{w})}}{Z}}
= \frac{e^{-E(\mathcal{v}, \mathcal{h}; \mathcal{w})}}{\sum_{h} e^{-E(\mathcal{v}, \mathcal{h}; \mathcal{w})}}
\\
\\
&
\mathbf{P}(h_{i}| v; \mathcal{w}) = \frac{e^{-E(v, h_{i}; w)}}{\sum_{h_{i}} e^{-E(v, h_{i}; w)}}
\\
\\
&
\Rightarrow \mathbf{P}(\mathcal{h}| \mathcal{v}; \mathcal{w}) = \frac{e^{-E(\mathcal{v}, \mathcal{h}; \mathcal{w})}}{\sum_{h} e^{-E(\mathcal{v}, \mathcal{h}; \mathcal{w})}}
= \frac{e^{- \sum_{i} E(v, h_{i}; w)}}{\sum_{h} e^{- \sum_{i} E(v, h_{i}; w)}}
= \frac{\prod_{i} e^{- E(v, h_{i}; w)}}{\sum_{h_{1}} \cdots \sum_{h_{d_{h}}} \prod_{i} e^{- E(v, h_{i}; w)}}\\
&
= \frac{\prod_{i} e^{- E(v, h_{i}; w)}}{\sum_{h_{1}} e^{- E(v, h_{1}; w)} \cdots \sum_{h_{d_{h}}} e^{- E(v, h_{d_{h}}; w)}} \\
&
=  \frac{\prod_{i} e^{- E(v, h_{i}; w)}} {\prod_{i} \sum_{h_{i}} e^{- E(v, h_{i}; w)}} 
\\
&
= \prod_{i} \frac{e^{- E(v, h_{i}; w)}} {\sum_{h_{i}} e^{- E(v, h_{i}; w)}}\\
&
= \prod_{i}{\mathbf{P}(h_{i}| v; \mathcal{w})}

\end{flalign*}
$$



Further more,
$$
\mathbf{P}(\mathcal{h}|\mathcal{v};\mathcal{w}) = \prod_{i}\mathbf{P}(\mathcal{h}_{i}|\mathcal{v};\mathcal{w}) \\

\mathbf{P}(\mathcal{v}|\mathcal{h}; \mathcal{w}) = \prod_{j}\mathbf{P}(\mathcal{v}_{j\cdot} | \mathcal{h}; \mathcal{w})
$$
### Propagation


For the condition probability,



$$
\begin{align*}

E(\mathcal{v}, \mathcal{h}_{i}; \mathcal{w})

&=  -\sum_{j, k} \mathcal{w}_{ijk}h_{i}v_{jk} 
\\
&= -\mathcal{h}_{i}\mathcal{w_{i}}\mathcal{v}
\\
\\

\mathbf{P}(\mathcal{h}_{i} = 1|\mathcal{v};\mathcal{w}) 

&= \frac{\mathbf{\mathbf{P}(\mathcal{h}_{i} = 1, \mathcal{v};\mathcal{w})}}{\mathbf{P}(\mathcal{h}_{i} = 0, \mathcal{v};\mathcal{w}) + \mathbf{P}(\mathcal{h}_{i} = 1, \mathcal{v};\mathcal{w})}
\\

&= \cdots
\\

&= \frac{e^{-E(\mathcal{v}, \mathcal{h_{i}}; \mathcal{w})}|_{\mathcal{h}_{i} = 1}}{e^{-E(\mathcal{v}, \mathcal{h_{i}}; \mathcal{w})}|_{\mathcal{h}_{i} = 0} + e^{-E(\mathcal{v}, \mathcal{h_{i}}; \mathcal{w})}|_{\mathcal{h}_{i} = 1}}
\\

&= \frac{e^{\sum_{j, k} \mathcal{w}_{ijk}v_{jk}}}{e^{0} + e^{\sum_{j, k} \mathcal{w}_{ijk}v_{jk}}}
\\

&=\frac{1}{e^{-\sum_{j, k} \mathcal{w}_{ijk}v_{jk}} + 1} \\

&= \sigma(\sum_{j, k} \mathcal{w}_{ijk}v_{jk})  \text{ where }  \sigma(x) = \frac{1}{1 + e^{-x}}
\\

& = \sigma(\sum_{j} \mathcal{w}_{ij\cdot} \cdot v_{j\cdot}), \text{ inner production on dimension k}


\end{align*}
$$



$$
\begin{align*}

\mathbf{P}(\mathcal{v}_{jk} = 1|\mathcal{h}; \mathcal{w}) 

& = \frac
{\mathbf{P}(\mathcal{v}_{jk} = 1, \mathcal{h}; \mathcal{w})}
{\sum_{k^{\prime} \neq k}\mathbf{P}(\mathcal{v}_{jk^{\prime}} = 1, \mathcal{v}_{j,-k^{\prime}} = 0, \mathcal{h}; \mathcal{w}) +\mathbf{P}(\mathcal{v}_{jk} = 1, \mathcal{h}; \mathcal{w})} \text{ since each  row of $\mathcal{v}$ is NOT independent, it's one-hot} \\

& = \cdots
\\

& = \frac
{
	e^{-E(\mathcal{v}_{jk} = 1, \mathcal{v}_{j, -k} = 0, \mathcal{h}; \mathcal{w})}
}
{
	\sum_{k_{\prime} \neq k} e^{-E(\mathcal{v}_{jk^{\prime}} = 1, \mathcal{v}_{j, -k^{\prime}} = 0, 		\mathcal{h}; \mathcal{w})}
	+
	e^{-E(\mathcal{v}_{jk} = 1, \mathcal{v}_{j, -k} = 0, \mathcal{h}; \mathcal{w})}
}
\\

& = \frac
{
	e^{\mathcal{h}^{T} \cdot \mathcal{w}_{\cdot jk}}
}
{
	\sum_{k^{\prime} \neq k}e^{\mathcal{h}^{T} \cdot \mathcal{w}_{\cdot jk^{\prime}}} + 							e^{\mathcal{h}^{T} \cdot \mathcal{w}_{\cdot jk}}
}
\\

& = Softmax(\sum_{i}\mathcal{h}_{i}\mathcal{w}_{ijk})
\text{ as project description's formula}


\end{align*}
$$

### Learning the weight

$$
\text{The obejctive is to minimize the KL divergence between $\mathbf{P}_{data}(\mathcal{v})$ and $\mathbf{P}(\mathcal{v}; \mathcal{w})$.}
\\
\text{The later is $\sum_{\mathcal{h}}\mathbf{P}(\mathcal{v}, \mathcal{h}; \mathcal{w})$, the marginal distribution of $\mathcal{v}$ from the joint distribution parameterized by $\mathcal{w}$}
$$



cited from "An Overview of Restricted Boltzmann Machines"


$$
\text{suppose we have N training samples, $\tau = \{\mathcal{v}^{(1)}, \cdots, \mathcal{v}^{(N)}\}$}
\\

\begin{align*}
d_{kl}(\mathbf{P}_{data}(v) || \mathbf{P}(\mathcal{v}; \mathcal{w}))

&= \sum_{v \in \tau} \mathbf{P}_{data}(v) \log(\frac{\mathbf{P}_{data}(v)}{\mathbf{P}(\mathcal{v}; \mathcal{w})})
\\

&= \sum_{v \in \tau} 
[ 
	\mathbf{P}_{data}(v) \log(\mathbf{P}_{data}(v)) - \mathbf{P}_{data}(v) \log(\mathbf{P}(\mathcal{v}; \mathcal{w}))
]
\end{align*}
$$

$$
\text{minimizing the KL divergence is equavilent to}
\\

\begin{align*}

\arg \min_{\mathcal{w}} d_{kl}(\mathbf{P}_{data}(v) || \mathbf{P}(\mathcal{v}; \mathcal{w}))

& = \arg \min_{\mathcal{w}}
\sum_{v \in \tau} 
[ 
	\mathbf{P}_{data}(v) \log(\mathbf{P}_{data}(v)) - \mathbf{P}_{data}(v) \log(\mathbf{P}(\mathcal{v}; \mathcal{w}))
]
\\

&= \arg \max_{\mathcal{w}} 
\sum_{v \in \tau} \mathbf{P}_{data}(\mathcal{v}) \log(\mathbf{P}(\mathcal{v}; \mathcal{w})
\\

&= \sum_{v \in \tau} \mathbf{P}_{data}(\mathcal{v})
\arg \max_{\mathcal{w}} \log(\mathbf{P}(\mathcal{v}; \mathcal{w}))
\
\text{when training}


\end{align*}
\\

\text{where $\mathbf{P}_{data}(\mathcal{v}) = \frac{1}{N}\sum_{i=1}^{N} \delta(\mathcal{v} - \mathcal{v}_{i})$, $\delta$ is the dirac delta function}
$$

$$
\begin{align*}

\ln \mathbf{P}(v; w) 

&= \ln \sum_{h} \mathbf{P}(v, h;w)
\\

&= \ln \sum_{h} \frac{1}{Z} e^{-E(v, h; w)} = \ln \frac{\sum_{h} e^{-E(v, h; w)}}{Z}
\\

&= \ln \sum_{h}e^{-E(v, h; w)} - \ln \sum_{v, h}e^{-E(v, h; w)}


\end{align*}
$$

Take gradient of each part w.r.t $w $

The former part
$$
\begin{align*}

\frac{\partial}{\partial w_{ijk}} \ln \sum_{h}e^{-E(v, h; w)} &=  \frac{1}{\sum_{h}e^{-E(v, h; w)}} \frac{\partial}{\partial w_{ijk}} \sum_{h}e^{-E(v, h; w)}

\\
&= \frac{1}{\sum_{h}e^{-E(v, h; w)}} \frac{\partial}{\partial w_{ijk}} \sum_{h}e^{-E(v, h; w)}
\\

&= \frac{1}{\sum_{h}e^{-E(v, h; w)}} \sum_{h} \frac{\partial}{\partial w_{ijk}} e^{-E(v, h; w)}
\\

&= \frac{1}{\sum_{h}e^{-E(v, h; w)}} \sum_{h} e^{-E(v, h;w)} \frac{\partial}{\partial w_{ijk}} (-E(v, h; w))
\\

&=- \frac{\sum_{h} e^{-E(v, h;w)} \frac{\partial}{\partial w_{ijk}} E(v, h; w)}{\sum_{h}e^{-E(v, h; w)}}
\\

&=- \sum_{h} \frac{e^{-E(v, h;w)}}{\sum_{h}e^{-E(v, h; w)}} \frac{\partial}{\partial w_{ijk}} E(v, h; w)
\\

&=- \sum_{h} \frac{\frac{e^{-E(v, h;w)}}{Z}}{\sum_{h}  \frac{e^{-E(v, h; w)}}{Z} } \frac{\partial}{\partial w_{ijk}} E(v, h; w)
\\

&= - \sum_{h}\mathbf{P}(h|v;w) \frac{\partial}{\partial w_{ijk}} E(v, h; w)
\\

&= \sum_{h}\mathbf{P}(h|v;w) \frac{\partial}{\partial w_{ijk}} \sum_{i, j, k} \mathcal{w}_{ijk}h_{i}v_{jk} 
\\

&= \sum_{h}\mathbf{P}(h|v;w) h_{i}v_{jk} 
\\

&= \sum_{h_{i} \in \{0, 1\}} \mathbf{P}(h_{i}|v;w) h_{i}v_{jk} \sum_{h_{-i}}\mathbf{P}(h_{i}|v;w) 
\\

&= \mathbf{P}(h_{i} = 1|v;w) v_{jk} \sum_{h_{-i}}\mathbf{P}(h_{i}|v;w)
\\

&= \mathbf{P}(h_{i} = 1|v;w) v_{jk} \
\text{
since 
$
\sum_{h_{-i}}\mathbf{P}(h_{i}|v;w) = \sum_{h_{i^{\prime}}}
\frac{ \prod_{i^{\prime} \neq i} \mathbf{P}(h_{i^{\prime}}|v;w)}{\sum_{h_{i^{\prime}}} \prod_{i^{\prime} \neq i} \mathbf{P}(h_{i^{\prime}}|v;w)}
= 1
$
}


\end{align*}
$$



The latter part


$$
\begin{align*}

\frac{\partial}{\partial w_{ijk}}  - \ln \sum_{v, h}e^{-E(v, h; w)} 
&= - \frac{1}{\sum_{v, h}e^{-E(v, h; w)}} \frac{\partial}{\partial w_{ijk}} \sum_{v, h}e^{-E(v, h; w)}
\\

&= - \frac{1}{\sum_{v, h}e^{-E(v, h; w)}} \sum_{v, h} e^{-E(v, h; w)} \frac{\partial}{\partial w_{ijk}} (-E(v, h; w))
\\

&= - \frac{1}{\sum_{v, h}e^{-E(v, h; w)}} \sum_{v, h} e^{-E(v, h; w)} \frac{\partial}{\partial w_{ijk}}
\sum_{i, j, k} \mathcal{w}_{ijk}h_{i}v_{jk}
\\

&= - \frac{1}{\sum_{v, h}e^{-E(v, h; w)}} \sum_{v, h} e^{-E(v, h; w)}h_{i}v_{jk}
\\

&= - \sum_{v, h} \frac{e^{-E(v, h; w)}}{Z} h_{i}v_{jk}
\\

&= - \sum_{v, h} \mathbf{P}(v, h; w) h_{i}v_{jk}
\\

&= - \sum_{v} \mathbf{P}(v; w) \textcolor{blue} {\sum_{h} \mathbf{P}(h|v;w)h_{i}v_{jk}}
\
\text{where the blue part is the same as the former}
\\

&= -\mathbf{E}_{v \sim \mathbf{P}(v;w)} [ \mathbf{P}(h_{i} = 1|v;w) v_{jk} ]
\\

&= -\mathbf{P}(h_{i} = 1|\bar{v};w) \bar{v}_{jk} \
\text{where $\bar{v}$ is the predicted $v$}

\end{align*}
$$

-----

## RBM with bias

### basic assumption

Now we introduce bias
$$
\begin{flalign*}

&
\mathcal{v} \in \{0, 1\}^{d_{v} \times d_{k}}, d_{v} \ \text{is the number of rating taregt,} d_{k} \ \text{ is the scale of rating}
&

\end{flalign*}
$$

$$
\begin{flalign*}

&
\mathcal{b^{v}} \in \mathbf{R}^{d_{v} \times d_{k}}\ \text{being the same shape as}\ \mathcal{v}
&

\end{flalign*}
$$

$$
\begin{flalign*}

&
b^{h} \in R^{d_{h}} \ \text{being the same shape as} \  h
&

\end{flalign*}
$$


$$
\begin{flalign*}

&

\mathcal{w} \in \mathcal{R}^{d_{h} \times d_{v} \times d_{k}}, d_{h}\ \text{is the number of hidden state}

&
\end{flalign*}
$$

$$
\begin{flalign*}
&
\mathcal{h} \in \{0, 1\}^{d_{h}}
&
\end{flalign*}
$$


$$
\begin{flalign*}

&
\text{For binary} \space{} \mathcal{v} \space{} \mathcal{h} \text{, define energy function:} \\

&
E(\mathcal{v}, \mathcal{h};\mathcal{w})
= - \sum_{ijk}w_{ijk}h_{i}v_{jk} - \sum_{jk} \tilde{b}^{v}_{jk}v_{jk} - \sum_{i} \tilde{b}^{h}_{i}h_{i}

\\
&
=  -\sum_{i, j, k} (\mathcal{w}_{ijk}h_{i}v_{jk} + \frac{1}{d_{h}} \tilde{b}^{v}_{jk}v_{jk} + \frac{1}{d_{v}d_{k}} \tilde{b}^{h}_{i}h_{i}) 
\\

&= -\sum_{i}\sum_{jk}(\mathcal{w}_{ijk}h_{i}v_{jk} + b^{v}_{jk}v_{jk} + b^{h}_{i}h_{i}) 
\space{}
\text{changing the notation to sclaing $b$}

\end{flalign*}
$$



$$
\mathbf{P}(\mathcal{v}, \mathcal{h}; \mathcal{w}) = 
\frac{e^{-E(\mathcal{v}, \mathcal{h}; \mathcal{w})}}{Z}
\space{}
\text{where Z is the marginal} \space{} \sum_{\mathcal{v}, \mathcal{h}}e^{-E(\mathcal{v}, \mathcal{h}; \mathcal{w})}
$$



$$
\begin{flalign*}
& 
\text{Note that $h_{i}$ are supposed to be independent for given $v$}.

&
\\
&
\Rightarrow \mathbf{P}(\mathcal{h}| \mathcal{v}; \mathcal{w}, b) = \prod_{i}\mathbf{P}(h_{i}| v; \mathcal{w}, b) 
\\
\\

&
\text{def } E(v, h_{i}; w, b) = -\sum_{j, k} (\mathcal{w}_{ijk}h_{i}v_{jk} + b^{v}_{jk}v_{jk} + b^{h}_{i}h_{i}) \\

&

i.e. \space{} \sum_{i} E(v, h_{i}; w, b) = E(v, h  ; w, b)

\\



&

\mathbf{P}(\mathcal{h}| \mathcal{v}; \mathcal{w}, b) 
= \frac{\mathbf{P}(\mathcal{h}, \mathcal{v}; \mathcal{w})}{\sum_{h} \mathbf{P}(\mathcal{h}, \mathcal{v}; \mathcal{w}) }
= \frac{\frac{e^{-E(\mathcal{v}, \mathcal{h}; \mathcal{w})}}{Z}}{\sum_{h}\frac{e^{-E(\mathcal{v}, \mathcal{h}; \mathcal{w})}}{Z}}
= \frac{e^{-E(\mathcal{v}, \mathcal{h}; \mathcal{w})}}{\sum_{h} e^{-E(\mathcal{v}, \mathcal{h}; \mathcal{w})}}
\\
\\
&
\mathbf{P}(h_{i}| v; \mathcal{w}, b) = \frac{e^{-E(v, h_{i}; w, b)}}{\sum_{h_{i}} e^{-E(v, h_{i}; w, b)}}
\\
\\
&
\Rightarrow \mathbf{P}(\mathcal{h}| \mathcal{v}; \mathcal{w}, b) = \frac{e^{-E(\mathcal{v}, \mathcal{h}; \mathcal{w}, b)}}{\sum_{h} e^{-E(\mathcal{v}, \mathcal{h}; \mathcal{w}, b)}}
= \frac{e^{- \sum_{i} E(v, h_{i}; w, b)}}{\sum_{h} e^{- \sum_{i} E(v, h_{i}; w, b)}}
= \frac{\prod_{i} e^{- E(v, h_{i}; w, b)}}{\sum_{h_{1}} \cdots \sum_{h_{d_{h}}} \prod_{i} e^{- E(v, h_{i}; w, b)}}\\
&
= \frac{\prod_{i} e^{- E(v, h_{i}; w, b)}}{\sum_{h_{1}} e^{- E(v, h_{1}; w, b)} \cdots \sum_{h_{d_{h}}} e^{- E(v, h_{d_{h}}; w, b)}} \\
&
=  \frac{\prod_{i} e^{- E(v, h_{i}; w, b)}} {\prod_{i} \sum_{h_{i}} e^{- E(v, h_{i}; w, b)}} 
\\
&
= \prod_{i} \frac{e^{- E(v, h_{i}; w, b)}} {\sum_{h_{i}} e^{- E(v, h_{i}; w, b)}}\\
&
= \prod_{i}{\mathbf{P}(h_{i}| v; \mathcal{w}, b)}

\end{flalign*}
$$



Further more,
$$
\mathbf{P}(\mathcal{h}|\mathcal{v};\mathcal{w}, b) = \prod_{i}\mathbf{P}(\mathcal{h}_{i}|\mathcal{v};\mathcal{w}, b) \\

\mathbf{P}(\mathcal{v}|\mathcal{h}; \mathcal{w}, b) = \prod_{j}\mathbf{P}(\mathcal{v}_{j\cdot} | \mathcal{h}; \mathcal{w}, b)
$$


### Propagation

For the condition probability,


$$
\begin{align*}

& \text{def marginal } E(\mathcal{v}, \mathcal{h}_{i}; \mathcal{w})
=  -\sum_{j, k} (\mathcal{w}_{ijk}h_{i}v_{jk} + b_{jk})
\\

&
\mathbf{P}(\mathcal{h}_{i} = 1|\mathcal{v};\mathcal{w}, b) 

= \frac{\mathbf{P}(\mathcal{h}_{i} = 1, \mathcal{v};\mathcal{w}, b)}{\mathbf{P}(\mathcal{h}_{i} = 0, \mathcal{v};\mathcal{w}, b) + \mathbf{P}(\mathcal{h}_{i} = 1, \mathcal{v};\mathcal{w}, b)}
\\

&= \cdots
\\

&= \frac{e^{-E(\mathcal{v}, \mathcal{h_{i}}; \mathcal{w}, b)}|_{\mathcal{h}_{i} = 1}}{e^{-E(\mathcal{v}, \mathcal{h_{i}}; \mathcal{w}, b)}|_{\mathcal{h}_{i} = 0} + e^{-E(\mathcal{v}, \mathcal{h_{i}}; \mathcal{w}, b)}|_{\mathcal{h}_{i} = 1}}
\\

&= \frac{e^{\sum_{j, k} (\mathcal{w}_{ijk}v_{jk} + b^{v}_{jk}v_{jk} + b^{h}_{i})}}{e^{\sum_{jk} b^{v}_{jk}v_{jk}} + e^{\sum_{j, k} (\mathcal{w}_{ijk}v_{jk} + b^{v}_{jk}v_{jk} + b^{h}_{i})}}
\\

&=\frac{1}{e^{-\sum_{j, k} (\mathcal{w}_{ijk}v_{jk} + b^{h}_{i})} + 1} \\

&= \sigma(\sum_{j, k} (\mathcal{w}_{ijk}v_{jk} + b^{h}_{i})) \space{} \text{where} \space{} \sigma(x) = \frac{1}{1 + e^{-x}}
\\

&= \sigma(\sum_{j, k} \mathcal{w}_{ijk}v_{jk} + \tilde{b}^{h}_{i}) \space{} \text{where} \space{} \sigma(x) = \frac{1}{1 + e^{-x}}
\\
&
\space{} \space{} \text{if we scale back $b$ to $\tilde{b}$ as Restricted Boltzmann Machines
for Collaborative Filtering's definition} 
\\

& = \sigma(\sum_{j} \mathcal{w}_{ij\cdot} \cdot v_{j\cdot} + \tilde{b}^{h}_{i}),\space{} \text{inner production on dimension k}


\end{align*}
$$



$$
\begin{align*}

&
E(v, h; w, b) = -\sum_{i}\sum_{jk}(\mathcal{w}_{ijk}h_{i}v_{jk} + b^{v}_{jk}v_{jk} + b^{h}_{i}h_{i}) \\

&
\text{def marginal }  E(v_{j \cdot}, h; w, b) = -\sum_{i}\sum_{k}(\mathcal{w}_{ijk}h_{i}v_{jk} + b^{v}_{jk}v_{jk} + b^{h}_{i}h_{i})  \\
& i.e. \space{} \sum_{j} E(v_{j \cdot}, h; w, b) = E(v, h; w, b)
\\
\\

& 
\mathbf{P}(v_{j\cdot}, h; w, b) = \sum_{v_{j^{\prime} \cdot}, j_{\prime} \neq j}P(v, h; w, b) 
\\

&
= \sum_{v_{j^{\prime} \cdot}, j^{\prime} \neq j} \frac{e^{-E(\mathcal{v}, \mathcal{h}; \mathcal{w})}}{Z}
\\

&
= \frac{\sum_{v_{j^{\prime} \cdot}, j^{\prime} \neq j} e^{-E(\mathcal{v}, \mathcal{h}; \mathcal{w})}} {Z}
\\

&
= \frac
	{
		e^{-E(v_{j \cdot}, h; w, b)} \prod_{j^{\prime} \neq j} (\sum_{v_{j^{\prime}}} e^{-					E(v_{j^{\prime} \cdot}, h; w, b)}) 
	}
	{
		Z 
	}
\\
\\

\end{align*}
$$

$$
\begin{align*}
&\text{def marginal } E(v_{jk}, h; w, b) = -\sum_{i}(\mathcal{w}_{ijk}h_{i}v_{jk} + b^{v}_{jk}v_{jk} + b^{h}_{i}h_{i})
\\

& i.e. \space{} \sum_{k} E(v_{jk}, h; w, b) = E(v_{j\cdot}, h;w, b)
\\

&
\mathbf{P}(v_{j\cdot}, h; w, b)
=  \frac
	{
		e^{-E(v_{j \cdot}, h; w, b)} \prod_{j^{\prime} \neq j} (\sum_{v_{j^{\prime}}} e^{-					E(v_{j^{\prime} \cdot}, h; w, b)}) 
	}
	{
		Z 
	}
= \frac
	{
		e^{-\sum_{k}E(v_{jk}, h; w, b)} \prod_{j^{\prime} \neq j} (\sum_{v_{j^{\prime}}} e^{- \sum_{k}					E(v_{j^{\prime} k}, h; w, b)}) 
	}
	{
		Z 
	}
\\

&
= e^{-\sum_{k}E(v_{jk}, h; w, b)}
\textcolor{blue}
{
\frac
	{
		\prod_{j^{\prime} \neq j} (\sum_{v_{j^{\prime}}} e^{- \sum_{k}					E(v_{j^{\prime} 		k}, h; w, b)}) 
	}
	{
		Z 
	}
}
\\

& \text{note the blue part is the same for all $v_{j \cdot} \in R^{d_{k}}$}
\\

&
\mathbf{P}(\mathcal{v}_{jk} = 1|\mathcal{h}; \mathcal{w}, b) \\

& = \frac
{\mathbf{P}(\mathcal{v}_{jk} = 1, \mathcal{h}; \mathcal{w})}
{\sum_{k^{\prime} \neq k}\mathbf{P}(\mathcal{v}_{jk^{\prime}} = 1, \mathcal{v}_{j,-k^{\prime}} = 0, \mathcal{h}; \mathcal{w}) +\mathbf{P}(\mathcal{v}_{jk} = 1, \mathcal{h}; \mathcal{w})} 
\space{}
\text{since each  row of $\mathcal{v}$ is NOT independent} \\

& =  
\frac
{
	e^
		{
			\sum_{i}(\mathcal{w}_{ijk}h_{i} + b^{v}_{jk} + b^{h}_{i}h_{i})
			+ \sum_{k^{\prime \prime} \neq k} \sum_{i}b^{h}_{i}h_{i}
		}
}
{
	\sum_{k^{\prime} \neq k}
    e^
    {
        \sum_{i}(\mathcal{w}_{ijk^{\prime}}h_{i} + b^{v}_{jk^{\prime}} + b^{h}_{i}h_{i})
        + \sum_{k^{\prime \prime} \neq k^{\prime}} \sum_{i}b^{h}_{i}h_{i}
    }
    + 
	e^
		{
			\sum_{i}(\mathcal{w}_{ijk}h_{i} + b^{v}_{jk} + b^{h}_{i}h_{i})
			+ \sum_{k^{\prime \prime} \neq k} \sum_{i}b^{h}_{i}h_{i}
		}
    
}
\\

& =
\frac
{
	e^
		{
			\sum_{i}(\mathcal{w}_{ijk}h_{i} + b^{v}_{jk})
			\textcolor{blue} 
			{+ \sum_{k^{\prime \prime} \in \{1, \cdots, d_{k} \}} \sum_{i}b^{h}_{i}h_{i}}
		}
}
{
	\sum_{k^{\prime} \neq k}
    e^
    {
        \sum_{i}(\mathcal{w}_{ijk^{\prime}}h_{i} + b^{v}_{jk^{\prime}})
        \textcolor{blue}
        { +\sum_{k^{\prime \prime} \in \{1, \cdots, d_{k} \} } \sum_{i}b^{h}_{i}h_{i}}
    }
    + 
	e^
		{
			\sum_{i}(\mathcal{w}_{ijk}h_{i} + b^{v}_{jk})
			\textcolor{blue}
			{+ \sum_{k^{\prime \prime} \in \{1, \cdots, d_{k} \}} \sum_{i}b^{h}_{i}h_{i}}
		}
    
}
\\

& =
\frac
{
	e^
		{
			\sum_{i}(\mathcal{w}_{ijk}h_{i} + b^{v}_{jk})
		}
}
{
	\sum_{k^{\prime} \neq k}
    e^
        {
        	\sum_{i}(\mathcal{w}_{ijk^{\prime}}h_{i} + b^{v}_{jk^{\prime}})
        }
	+
    e^
		{
			\sum_{i}(\mathcal{w}_{ijk}h_{i} + b^{v}_{jk})
		}
	
} \text{ after cancelling the blue part}
\\

& = 
\frac
{
	e^
		{
			\sum_{i}\mathcal{w}_{ijk}h_{i} + \tilde{b}^{v}_{jk}
		}
}
{
	\sum_{k^{\prime} \neq k}
    e^
        {
        	\sum_{i}\mathcal{w}_{ijk^{\prime}}h_{i} + \tilde{b}^{v}_{jk^{\prime}}
        }
	+
    e^
		{
			\sum_{i}\mathcal{w}_{ijk}h_{i} + \tilde{b}^{v}_{jk}
		}
	
} \text{ if we scale $b$ back}

\end{align*}
$$

