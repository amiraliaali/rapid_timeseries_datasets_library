% ===
%
% Official LaTeX seminar report class of the
% Chair for AI Methodology (AIM)
% RWTH Aachen University, Aachen, Germany
%
% Author: Jakob Bossek (bossek@aim.rwth-aachen.de)
%
% AIM website: https://aim.rwth-aachen.de/
%
% NOTE: DO NOT EDIT THIS FILE!
% ===

% set LaTeX version to work with
\NeedsTeXFormat{LaTeX2e}

% declare class
\ProvidesClass{AIM_report}[2022/04/12 RWTH Aachen, Chair for AI Methodology seminar report template]

% Preliminary declarations
% ===

% Line numbering for review mode
\usepackage[switch,running,mathlines]{lineno}


% take care of all options not defined explicitly
\DeclareOption*
{\PassOptionsToClass{\currentOption}{llncs}}

% review option activates line numbering (note that math environments \align and \equation are not numbered)
\DeclareOption{review}{\linenumbers}
\ProcessOptions\relax
% template extends base article class
\LoadClass[10pt, runningheads]{llncs}

% More declarations
% ===

% NOTE: \RequirePackage is like \usepackage, but it can be called before \documentclass

\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}

% Margins
\RequirePackage[a4paper,left=2.9cm, top=2.9cm, bottom=2.3cm, right=2.9cm,heightrounded,includefoot]{geometry}

% floats
\RequirePackage{float}
\RequirePackage[font=small,labelfont=bf,justification=justified]{caption}
\RequirePackage{subcaption}
%\RequirePackage[caption=false]{subfig}

% font
\RequirePackage{palatino}
\usepackage{microtype} % minor typographic corrections
\RequirePackage{mathpazo} % palatino math font

% math
\RequirePackage{amsmath}
\RequirePackage{amssymb}
%\RequirePackage{amsthm}
\RequirePackage{mathtools}
\RequirePackage{dsfont}

\RequirePackage{graphicx}

% colors
\RequirePackage[dvipsnames]{xcolor}

% define some colors
\definecolor{rwthlightblue}{RGB}{143, 185, 223}
\definecolor{rwthdarkblue}{RGB}{0, 84, 159}

% algorithms
\RequirePackage{algorithm}
\RequirePackage{algpseudocode}

% bibliography
\RequirePackage[backend=biber,sorting=none,citestyle=alphabetic,style=alphabetic]{biblatex}

% hyperlinks
\RequirePackage{hyperref}
\hypersetup{colorlinks,linkcolor={rwthdarkblue},citecolor={rwthdarkblue},urlcolor={rwthdarkblue}}

% Header and Footer
\renewcommand{\figurename}{Figure}
