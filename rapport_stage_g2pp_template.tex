\documentclass[11pt,a4paper]{article}

%========================================================
% Packages sobres, compatibles avec un environnement verrouillé
%========================================================
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[french]{babel}
\usepackage{lmodern}
\usepackage{geometry}
\usepackage{microtype}
\usepackage{amsmath,amssymb,mathtools}
\usepackage{booktabs,array,longtable}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric,calc}
\usepackage[hidelinks]{hyperref}

%========================================================
% Mise en page
%========================================================
\geometry{left=2.4cm,right=2.4cm,top=2.5cm,bottom=2.5cm}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.55em}
\setlist[itemize]{leftmargin=1.2em,itemsep=0.15em,topsep=0.2em}
\setlist[enumerate]{leftmargin=1.4em,itemsep=0.15em,topsep=0.2em}

%========================================================
% Couleurs et macros
%========================================================
\definecolor{LBPBlue}{RGB}{27,64,115}
\definecolor{SoftGray}{RGB}{245,246,248}
\definecolor{DarkGray}{RGB}{70,70,70}
\definecolor{Accent}{RGB}{0,120,130}

\newcommand{\Gpp}{G2\texttt{++}}
\newcommand{\Gopp}{G1\texttt{++}}
\newcommand{\dd}{\,\mathrm{d}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\Var}{\operatorname{Var}}
\newcommand{\Cov}{\operatorname{Cov}}
\newcommand{\placeholder}[1]{\textcolor{DarkGray}{\emph{#1}}}
\newcommand{\code}[1]{\texttt{#1}}

\newenvironment{reportbox}[1]
{%
\vspace{0.3em}
\noindent\fcolorbox{LBPBlue}{SoftGray}{%
\begin{minipage}{0.96\linewidth}
\textbf{#1}\par\vspace{0.2em}
}
{%
\end{minipage}}
\vspace{0.3em}
}

%========================================================
% TikZ styles
%========================================================
\tikzset{
    block/.style={draw=LBPBlue, rounded corners, very thick, align=center,
        fill=SoftGray, text width=3.2cm, minimum height=1.0cm, inner sep=5pt},
    smallblock/.style={draw=LBPBlue, rounded corners, thick, align=center,
        fill=white, text width=2.8cm, minimum height=0.85cm, inner sep=4pt},
    planned/.style={draw=Accent, rounded corners, thick, dashed, align=center,
        fill=white, text width=3.0cm, minimum height=0.85cm, inner sep=4pt},
    arrow/.style={-{Latex[length=2.5mm]}, thick, draw=DarkGray},
    softarrow/.style={-{Latex[length=2.5mm]}, dashed, thick, draw=Accent}
}

%========================================================
% Métadonnées
%========================================================
\title{Rapport d'avancement --- Construction et calibration d'un modèle \Gpp}
\author{\placeholder{Nom de l'auteur}\\
\placeholder{Équipe / Direction / Entreprise}}
\date{\placeholder{Date de rédaction}}

\begin{document}

\maketitle

\begin{abstract}
Ce document présente l'état d'avancement de la construction d'un modèle de taux court gaussien à deux facteurs, \Gpp, au sein d'une librairie de pricing existante. L'objectif immédiat est de documenter les résultats d'erreur et d'instabilité observés sur les deux premiers maîtres de calibration de la procédure inspirée de Bonino et al.\ (2020) : une calibration \Gopp\ globale, puis une calibration \Gopp\ par bucket et par tenor. La calibration \Gpp\ globale constitue l'étape suivante de l'architecture, mais ses résultats numériques ne sont pas encore présentés dans cette version.
\end{abstract}

\tableofcontents
\newpage

%========================================================
\section{Contexte et objet du document}
%========================================================

Le projet s'inscrit dans le développement d'un modèle de taux court \Gpp\ destiné à être intégré dans une librairie de pricing de production. L'enjeu n'est pas seulement de reproduire des formules fermées de valorisation de swaptions européennes, mais de construire une implémentation stable, traçable et compatible avec les objets existants de la librairie : courbes zéro-coupon, conventions de dates, instruments de taux, surfaces ou cubes de volatilité, moteurs de calibration et diagnostics numériques.

Le modèle cible est le modèle gaussien à deux facteurs, défini par
\begin{equation}
    r_t = x_t + y_t + \varphi(t),
\end{equation}
où les deux facteurs stochastiques suivent des dynamiques d'Ornstein--Uhlenbeck sous la mesure risque-neutre :
\begin{equation}
\left\{
\begin{aligned}
    \dd x_t &= -a x_t\dd t + \sigma(t)\dd W_t^x, \qquad x_0=0,\\
    \dd y_t &= -b y_t\dd t + \eta(t)\dd W_t^y, \qquad y_0=0,\\
    \dd\langle W^x,W^y\rangle_t &= \rho\dd t.
\end{aligned}
\right.
\end{equation}
La fonction déterministe $\varphi$ assure la reconstitution de la courbe initiale. Les paramètres $a,b>0$ contrôlent les vitesses de retour à la moyenne, tandis que $\sigma(t)$ et $\eta(t)$ portent la structure de volatilité. Dans l'implémentation visée, ces volatilités sont supposées constantes par morceaux sur une partition de temps commune :
\begin{equation}
    \sigma(t)=\sum_{k=1}^{K}\sigma_k\mathbf 1_{(\tau_{k-1},\tau_k]}(t),
    \qquad
    \eta(t)=\sum_{k=1}^{K}\eta_k\mathbf 1_{(\tau_{k-1},\tau_k]}(t).
\end{equation}

Le présent rapport ne couvre pas encore la calibration complète du modèle \Gpp. Il se concentre sur les deux premières briques de calibration utilisées comme étape de stabilisation :
\begin{enumerate}
    \item une calibration \Gopp\ globale, où un premier jeu de paramètres est estimé sur une grille de swaptions ;
    \item une calibration \Gopp\ par bucket et par tenor, destinée à tester la stabilité locale de la structure de volatilité et la sensibilité de l'erreur aux découpages de maturité.
\end{enumerate}
Ces deux étapes ont une fonction de contrôle : elles permettent d'identifier les zones de surface où le schéma de calibration devient instable avant d'introduire la calibration \Gpp\ globale.

\begin{reportbox}{Objet opérationnel du rapport}
Documenter la structure du modèle, son intégration dans la librairie de pricing, puis présenter les métriques d'erreur et de stabilité obtenues sur les deux premiers maîtres de calibration : \Gopp\ global et \Gopp\ par bucket/tenor. Les résultats \Gpp\ globaux restent hors périmètre dans cette version.
\end{reportbox}

%========================================================
\subsection{Schéma général de calibration}
%========================================================

La chaîne de calibration suit une logique progressive. Le modèle \Gopp\ joue ici un rôle de réduction contrôlée : il permet de tester la cohérence des conventions, des instruments et des fonctions objectif avant d'activer le modèle \Gpp\ complet.

\begin{figure}[h!]
\centering
\begin{tikzpicture}[node distance=1.2cm and 1.4cm]
    \node[block] (market) {Données marché\\ courbes + cube swaption};
    \node[block, right=of market] (pack) {Construction instruments\\ expiries, tenors, strikes};
    \node[block, right=of pack] (g1global) {Maître 1\\ \Gopp\ global};
    \node[block, below=of g1global] (g1local) {Maître 2\\ \Gopp\ par bucket/tenor};
    \node[block, left=of g1local] (diag) {Diagnostics\\ erreurs, stabilité, heatmaps};
    \node[planned, right=of g1local] (g2global) {Étape suivante\\ \Gpp\ global\\ non présenté};

    \draw[arrow] (market) -- (pack);
    \draw[arrow] (pack) -- (g1global);
    \draw[arrow] (g1global) -- (g1local);
    \draw[arrow] (g1local) -- (diag);
    \draw[softarrow] (g1local) -- (g2global);
\end{tikzpicture}
\caption{Pipeline de calibration documenté dans ce rapport.}
\label{fig:pipeline-calibration}
\end{figure}

%========================================================
\section{Modèle, conventions et responsabilités mathématiques}
%========================================================

Le modèle \Gpp\ repose sur une représentation affine du prix zéro-coupon. Pour $0\leq t\leq T$, on utilise une écriture de la forme
\begin{equation}
    P(t,T)=A(t,T)\exp\bigl(-B_x(t,T)x_t-B_y(t,T)y_t\bigr),
\end{equation}
où
\begin{equation}
    B_x(t,T)=\frac{1-e^{-a(T-t)}}{a},
    \qquad
    B_y(t,T)=\frac{1-e^{-b(T-t)}}{b}.
\end{equation}
Le terme $A(t,T)$ incorpore la courbe initiale et les corrections de variance nécessaires à la reconstitution de la structure zéro-coupon observée.

La quantité centrale pour les tests de cohérence est la variance intégrée de la partie stochastique du taux court :
\begin{equation}
    \Gamma(s,t)=\Var\left(\int_s^t x_u+y_u\,\dd u\right).
\end{equation}
Dans le cas de volatilités constantes par morceaux, cette variance est calculée comme une somme de contributions locales sur les buckets actifs :
\begin{equation}
    \Gamma(s,t)=\sum_{k\in\mathcal K(s,t)}
    \left(\Gamma_x^{(k)}(s,t)+\Gamma_y^{(k)}(s,t)+\Gamma_{xy}^{(k)}(s,t)\right).
\end{equation}
Cette décomposition est importante pour la calibration par bucket : une erreur d'indexation sur les bornes locales modifie directement les prix, les volatilités implicites et les résidus.

\begin{reportbox}{Responsabilité mathématique}
La couche modèle doit garantir la cohérence entre quatre objets : la courbe initiale, les coefficients affines $A,B_x,B_y$, les moments gaussiens sous la mesure forward pertinente, et la fonction objectif utilisée en calibration. Les calibrations \Gopp\ servent ici de tests de réduction avant la calibration \Gpp.
\end{reportbox}

%========================================================
\subsection{Réduction \Gopp\ utilisée dans les deux premiers maîtres}
%========================================================

Les deux premiers maîtres de calibration reposent sur une réduction à un facteur. On considère alors une dynamique de type
\begin{equation}
\left\{
\begin{aligned}
    r_t^{(1)} &= x_t+\varphi(t),\\
    \dd x_t &= -a x_t\dd t+\sigma(t)\dd W_t.
\end{aligned}
\right.
\end{equation}
La calibration \Gopp\ globale estime un paramétrage unique sur un ensemble d'instruments. La calibration par bucket et par tenor relâche ensuite cette structure pour analyser la stabilité locale des erreurs.

%========================================================
\section{Héritage et intégration dans la librairie de pricing}
%========================================================

L'intégration du modèle dans la librairie doit préserver une séparation stricte entre les responsabilités. Le modèle ne doit pas reconstruire implicitement les instruments à l'intérieur des noyaux numériques ; inversement, les instruments ne doivent pas contenir de logique de calibration ou d'accès caché aux paramètres du modèle.

\begin{figure}[h!]
\centering
\begin{tikzpicture}[node distance=0.9cm and 1.2cm]
    \node[smallblock] (models) {Classe abstraite\\ \code{Models}};
    \node[smallblock, below=of models] (g2) {Modèle\\ \code{GaussianTwoFactors}};
    \node[smallblock, left=of g2] (dcf) {\code{DCFModel}\\ discount / forecast};
    \node[smallblock, left=of dcf] (zc) {\code{ZCCurve}\\ courbes ZC};
    \node[smallblock, right=of g2] (instr) {Instruments\\ IRS / swaptions};
    \node[smallblock, right=of instr] (cube) {Surface / cube\\ volatilités};
    \node[smallblock, below=of g2] (calib) {Calibration\\ objectifs + diagnostics};

    \draw[arrow] (models) -- (g2);
    \draw[arrow] (zc) -- (dcf);
    \draw[arrow] (dcf) -- (g2);
    \draw[arrow] (cube) -- (instr);
    \draw[arrow] (instr) -- (g2);
    \draw[arrow] (g2) -- (calib);
\end{tikzpicture}
\caption{Intégration fonctionnelle du modèle dans la librairie.}
\label{fig:integration-librairie}
\end{figure}

\subsection{Objets principaux}

\begin{longtable}{p{0.28\linewidth}p{0.64\linewidth}}
\toprule
\textbf{Objet} & \textbf{Responsabilité attendue}\\
\midrule
\code{ZCCurve} & Fournir les facteurs d'actualisation initiaux et les taux forwards nécessaires au recalage de la courbe.\\
\code{DCFModel} & Centraliser les courbes discount et forecast utilisées par le modèle.\\
\code{GaussianTwoFactors} & Porter les paramètres du modèle, les formules affines, les variances intégrées et les moments gaussiens.\\
\code{IRSVanille} & Construire les dates de paiement, accruals et conventions de la jambe fixe sous-jacente.\\
\code{SwaptionCube} / \code{SwaptionSurface} & Fournir les quotes de marché organisées par expiry, tenor et éventuellement strike.\\
Maître de calibration & Transformer les paramètres, lancer l'optimiseur, stocker les résidus et produire les diagnostics.\\
\bottomrule
\end{longtable}

\subsection{Convention d'architecture}

La convention retenue est la suivante : les méthodes bas niveau reçoivent uniquement les données numériques nécessaires. Elles ne doivent pas accéder directement au cube de volatilité, reconstruire un instrument, ni modifier l'état du modèle. La logique de calibration doit rester dans une couche séparée.

\begin{reportbox}{Règle d'intégration}
Une méthode de pricing peut appeler les formules du modèle et les spécifications d'instrument déjà préparées. Une fonction objectif de calibration peut appeler le pricing. En revanche, un noyau de variance, de covariance ou de moment ne doit pas déclencher une reconstruction d'instrument ou une recalibration implicite.
\end{reportbox}

%========================================================
\section{Structure des maîtres de calibration}
%========================================================

On note $q_{ij}^{\mathrm{mkt}}$ la quote de marché associée à l'expiry $T_i$ et au tenor $S_j$, et $q_{ij}^{\mathrm{mod}}(\theta)$ la quote reconstruite par le modèle pour un vecteur de paramètres $\theta$. Une fonction objectif générique s'écrit
\begin{equation}
    \theta^\star
    =\arg\min_{\theta\in\Theta}
    \sum_{(i,j)\in\mathcal I}w_{ij}
    \left(q_{ij}^{\mathrm{mod}}(\theta)-q_{ij}^{\mathrm{mkt}}\right)^2.
\end{equation}
Lorsque les prix de swaptions sont utilisés directement, on peut normaliser par l'annuité ou par une échelle de prix afin d'éviter qu'une zone de la surface domine mécaniquement la calibration :
\begin{equation}
    e_{ij}^{\mathrm{norm}}(\theta)
    =\frac{\Pi_{ij}^{\mathrm{mod}}(\theta)-\Pi_{ij}^{\mathrm{mkt}}}
    {\max(A_{ij},\varepsilon)}.
\end{equation}

\subsection{Maître 1 : calibration \Gopp\ globale}

La calibration \Gopp\ globale sert de premier test de cohérence. Elle impose un paramétrage unique sur tout l'ensemble de calibration. Elle permet de répondre à trois questions :
\begin{enumerate}
    \item le moteur de pricing renvoie-t-il des valeurs finies et positives sur toute la grille ?
    \item les résidus présentent-ils une structure systématique par expiry ou par tenor ?
    \item les paramètres estimés restent-ils dans une région économiquement et numériquement admissible ?
\end{enumerate}

\subsection{Maître 2 : calibration \Gopp\ par bucket et par tenor}

La calibration par bucket et par tenor relâche la contrainte globale. Elle vise à isoler les zones instables de la surface. Le diagnostic principal n'est pas seulement le niveau d'erreur final, mais la régularité des paramètres estimés et la propagation des résidus d'un bucket au suivant.

\begin{equation}
    \theta_{k,j}^\star
    =\arg\min_{\theta_{k,j}\in\Theta_{k,j}}
    \sum_{i\in\mathcal I(k,j)}w_{ij}
    \left(q_{ij}^{\mathrm{mod}}(\theta_{k,j})-q_{ij}^{\mathrm{mkt}}\right)^2.
\end{equation}
Ici, $k$ désigne le bucket de calibration et $j$ le tenor. La notation $\mathcal I(k,j)$ représente le sous-ensemble d'instruments actifs dans la calibration locale.

\subsection{Maître 3 : calibration \Gpp\ globale}

La calibration \Gpp\ globale correspond à l'étape suivante. Elle n'est pas présentée numériquement dans ce rapport. Elle utilisera les diagnostics précédents comme garde-fous : régions instables, choix de bornes, poids de calibration, initialisation et conventions de transformation des paramètres.

%========================================================
\section{Métriques d'erreur et diagnostics de stabilité}
%========================================================

Les diagnostics doivent séparer les erreurs de prix, les erreurs de volatilité implicite et les instabilités de paramètres. Les métriques principales sont les suivantes :
\begin{align}
    \mathrm{RMSE} &= \sqrt{\frac{1}{N}\sum_{(i,j)\in\mathcal I}e_{ij}^2},\\
    \mathrm{MAE}  &= \frac{1}{N}\sum_{(i,j)\in\mathcal I}|e_{ij}|,\\
    \mathrm{MaxAE} &= \max_{(i,j)\in\mathcal I}|e_{ij}|.
\end{align}
Pour les volatilités, on distingue l'erreur absolue en points de volatilité et l'erreur relative :
\begin{equation}
    \Delta\sigma_{ij}=\sigma_{ij}^{\mathrm{mod}}-\sigma_{ij}^{\mathrm{mkt}},
    \qquad
    \Delta\sigma_{ij}^{\%}=100\times
    \frac{\sigma_{ij}^{\mathrm{mod}}-\sigma_{ij}^{\mathrm{mkt}}}
    {\sigma_{ij}^{\mathrm{mkt}}}.
\end{equation}

\begin{reportbox}{Lecture des erreurs}
Une faible erreur moyenne ne suffit pas. Pour qualifier la stabilité de la calibration, il faut vérifier la localisation des erreurs, la présence de structures par expiry/tenor, la sensibilité aux conditions initiales et la régularité des paramètres calibrés d'un bucket au suivant.
\end{reportbox}

%========================================================
\section{Résultats numériques}
%========================================================

Cette section est volontairement structurée comme un espace de remplissage. Les tableaux peuvent être complétés avec les sorties MATLAB des deux premiers maîtres de calibration.

\subsection{Résultats du maître 1 : \Gopp\ global}

\begin{table}[h!]
\centering
\begin{tabular}{lrrrr}
\toprule
\textbf{Bloc} & \textbf{RMSE} & \textbf{MAE} & \textbf{MaxAE} & \textbf{Statut}\\
\midrule
Prix normalisés & \placeholder{--} & \placeholder{--} & \placeholder{--} & \placeholder{À compléter}\\
Volatilités implicites & \placeholder{--} & \placeholder{--} & \placeholder{--} & \placeholder{À compléter}\\
Résidus pondérés & \placeholder{--} & \placeholder{--} & \placeholder{--} & \placeholder{À compléter}\\
\bottomrule
\end{tabular}
\caption{Synthèse des erreurs pour la calibration \Gopp\ globale.}
\label{tab:g1-global-errors}
\end{table}

\begin{table}[h!]
\centering
\begin{tabular}{lrrr}
\toprule
\textbf{Paramètre} & \textbf{Valeur initiale} & \textbf{Valeur calibrée} & \textbf{Commentaire}\\
\midrule
$a$ & \placeholder{--} & \placeholder{--} & \placeholder{--}\\
$\sigma$ & \placeholder{--} & \placeholder{--} & \placeholder{--}\\
\bottomrule
\end{tabular}
\caption{Paramètres de la calibration \Gopp\ globale.}
\label{tab:g1-global-params}
\end{table}

\subsection{Résultats du maître 2 : \Gopp\ par bucket et par tenor}

\begin{table}[h!]
\centering
\begin{tabular}{lrrrrr}
\toprule
\textbf{Bucket} & \textbf{Tenor} & \textbf{RMSE} & \textbf{MaxAE} & \textbf{Convergence} & \textbf{Commentaire}\\
\midrule
\placeholder{B1} & \placeholder{1Y}  & \placeholder{--} & \placeholder{--} & \placeholder{--} & \placeholder{--}\\
\placeholder{B1} & \placeholder{5Y}  & \placeholder{--} & \placeholder{--} & \placeholder{--} & \placeholder{--}\\
\placeholder{B2} & \placeholder{10Y} & \placeholder{--} & \placeholder{--} & \placeholder{--} & \placeholder{--}\\
\bottomrule
\end{tabular}
\caption{Synthèse locale des erreurs par bucket et par tenor.}
\label{tab:g1-bucket-tenor-errors}
\end{table}

\subsection{Figures diagnostiques à insérer}

\begin{figure}[h!]
\centering
\fbox{\begin{minipage}[c][5.0cm][c]{0.86\linewidth}
\centering
\placeholder{Insérer ici la heatmap des erreurs de volatilité implicite : expiry $\times$ tenor.}
\end{minipage}}
\caption{Carte d'erreur de volatilité implicite.}
\label{fig:heatmap-vol-error}
\end{figure}

\begin{figure}[h!]
\centering
\fbox{\begin{minipage}[c][5.0cm][c]{0.86\linewidth}
\centering
\placeholder{Insérer ici le graphe de stabilité des paramètres par bucket.}
\end{minipage}}
\caption{Stabilité des paramètres calibrés par bucket.}
\label{fig:param-stability}
\end{figure}

%========================================================
\section{Discussion provisoire}
%========================================================

Les calibrations \Gopp\ ont une fonction de diagnostic. La calibration globale permet de vérifier la cohérence de la chaîne marché--instrument--pricing--objectif. La calibration par bucket et par tenor permet ensuite d'identifier les zones où la surface impose une flexibilité locale importante ou produit des paramètres peu réguliers. Ces observations conditionnent directement la calibration \Gpp\ globale : choix des bornes, transformation des paramètres, initialisation, pondération des instruments et filtrage éventuel des points instables.

\begin{reportbox}{Conclusion intermédiaire}
Le passage vers \Gpp\ ne doit pas être traité comme une simple augmentation du nombre de paramètres. Il doit s'appuyer sur les diagnostics \Gopp\ : résidus structurés, instabilités locales, sensibilité aux buckets et robustesse de la chaîne d'intégration dans la librairie.
\end{reportbox}

%========================================================
\appendix
\section{Annexe A --- Checklist de validation}
%========================================================

\begin{enumerate}
    \item Vérifier que tous les instruments de calibration sont construits avec les mêmes conventions de dates et d'accruals.
    \item Vérifier que les prix modèles sont finis, positifs et compatibles avec les bornes intrinsèques.
    \item Vérifier que les volatilités implicites reconstruites existent sur toute la grille utilisée.
    \item Vérifier que les résidus ne sont pas dominés par une seule zone de la surface.
    \item Vérifier que la calibration par bucket ne produit pas de sauts artificiels de paramètres.
    \item Vérifier que les paramètres utilisés pour initialiser \Gpp\ sont admissibles : $a,b,\sigma,\eta>0$ et $|\rho|<1$.
\end{enumerate}

%========================================================
\section{Annexe B --- Références internes aux méthodes}
%========================================================

\begin{longtable}{p{0.32\linewidth}p{0.58\linewidth}}
\toprule
\textbf{Méthode / objet} & \textbf{Rôle dans le rapport}\\
\midrule
\code{getBucketBounds} & Construction des bornes globales des buckets de volatilité.\\
\code{getPricingIdx} & Identification des buckets actifs pour un intervalle $[s,t]$.\\
\code{vari}, \code{variX}, \code{variY}, \code{variXY} & Calcul de la variance intégrée et de ses contributions.\\
\code{getBondPrice} & Reconstitution affine du prix zéro-coupon.\\
\code{getSwaptionBondBasket} & Construction des coefficients affines du panier obligataire sous-jacent.\\
\code{priceSwaptionG2PP} & Pricing cible pour la calibration \Gpp\ globale future.\\
\bottomrule
\end{longtable}

%========================================================
\begin{thebibliography}{9}
%========================================================

\bibitem{bonino2020}
Bonino et al.\ (2020).
\emph{Référence exacte à compléter : procédure de calibration utilisée pour les modèles de taux gaussiens à volatilité constante par morceaux.}

\bibitem{brigoMercurio}
Brigo, D. et Mercurio, F.
\emph{Interest Rate Models -- Theory and Practice}.
Springer.

\end{thebibliography}

\end{document}
