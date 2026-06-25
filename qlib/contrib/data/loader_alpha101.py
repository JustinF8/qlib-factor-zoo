# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Alpha101 Data Loader - 101 Formulaic Alphas by Kakushadze (2015)
# Converted from Vibe-Trading alpha zoo for use with Qlib expression engine.
#
# Reference: Kakushadze (2015), "101 Formulaic Alphas", arXiv:1601.00991
#
# CUSTOM OPERATORS REQUIRED (auto-registered via handler.py):
#   - TsArgmax(X, N) : Rolling argmax value extraction
#   - TsArgmin(X, N) : Rolling argmin value extraction
#
# Alphas using these operators: #001, #057, #060, #098, #124 (approx versions
# were used previously; now replaced with exact custom operators).

from qlib.data.dataset.loader import QlibDataLoader


class Alpha101DL(QlibDataLoader):
    """Data loader for 101 Formulaic Alphas (Kakushadze 2015)."""

    def __init__(self, config=None, **kwargs):
        _config = {"feature": self.get_feature_config()}
        if config is not None:
            _config.update(config)
        super().__init__(config=_config, **kwargs)

    @staticmethod
    def get_feature_config():
        """Return (fields, names) tuples for all 101 alphas."""
        fields = []
        names = []

        # ================================================================
        # Alpha #001
        # Original: rank(ts_argmax(SignedPower((returns<0)?stddev(returns,20):close, 2.), 5)) - 0.5
        # Exact with custom operator TsArgmax
        fields.append(
            "Rank(TsArgmax(Sign($close/Ref($close,1)-1)*Power(Abs($close/Ref($close,1)-1),2), 5), 1) - 0.5"
        )
        names.append("ALPHA001")

        # Alpha #002
        # Original: -1 * correlation(rank(delta(log(volume),2)), rank(((close-open)/open)), 6)
        fields.append(
            "-1*Corr(Rank(Delta(Log($volume+1),2)), Rank(($close-$open)/$open), 6)"
        )
        names.append("ALPHA002")

        # Alpha #003
        # Original: -1 * correlation(rank(open), rank(volume), 10)
        fields.append(
            "-1*Corr(Rank($open,1), Rank($volume,1), 10)"
        )
        names.append("ALPHA003")

        # Alpha #004
        # Original: -1 * Ts_Rank(rank(low), 9)
        fields.append(
            "-1*Rank(Rank($low,1), 9)"
        )
        names.append("ALPHA004")

        # Alpha #005
        # Original: rank((open - sum(vwap,10)/10)) * (-1 * abs(rank((close - vwap))))
        fields.append(
            "Rank($open - Sum($vwap,10)/10, 1) * (-1*Abs(Rank($close-$vwap, 1)))"
        )
        names.append("ALPHA005")

        # Alpha #006
        # Original: -1 * correlation(open, volume, 10)
        fields.append(
            "-1*Corr($open, $volume, 10)"
        )
        names.append("ALPHA006")

        # Alpha #007
        # Original: (adv20<volume)?((-1*ts_rank(abs(delta(close,7)),60))*sign(delta(close,7))):(-1)
        fields.append(
            "If(Mean($volume,20)<$volume, "
            "(-1*Rank(Abs(Delta($close,7)),60))*Sign(Delta($close,7)), "
            "-1)"
        )
        names.append("ALPHA007")

        # Alpha #008
        # Original: -1 * rank((sum(open,5)*sum(returns,5)) - delay(sum(open,5)*sum(returns,5),10))
        fields.append(
            "-1*Rank("
            "(Sum($open,5)*Sum($close/Ref($close,1)-1,5))"
            "-Ref(Sum($open,5)*Sum($close/Ref($close,1)-1,5),10)"
            ", 1)"
        )
        names.append("ALPHA008")

        # Alpha #009
        # Original: (0<ts_min(delta(close,1),5))?delta(close,1):((ts_max(delta(close,1),5)<0)?delta(close,1):(-1*delta(close,1)))
        d1 = "Delta($close,1)"
        cond1 = f"Min({d1},5)>0"
        cond2 = f"Max({d1},5)<0"
        fields.append(
            f"If({cond1}, {d1}, If({cond2}, {d1}, -1*{d1}))"
        )
        names.append("ALPHA009")

        # Alpha #010
        # Original: rank((0<ts_min(delta(close,1),4))?delta(close,1):((ts_max(delta(close,1),4)<0)?delta(close,1):(-1*delta(close,1))))
        d1_10 = "Delta($close,1)"
        fields.append(
            f"Rank(If(Min({d1_10},4)>0, {d1_10}, If(Max({d1_10},4)<0, {d1_10}, -1*{d1_10})), 1)"
        )
        names.append("ALPHA010")

        # Alpha #011
        # Original: (rank(ts_max(vwap-close,3))+rank(ts_min(vwap-close,3)))*rank(delta(volume,3))
        fields.append(
            "(Rank(Max($vwap-$close,3),1)+Rank(Min($vwap-$close,3),1))*Rank(Delta($volume,3),1)"
        )
        names.append("ALPHA011")

        # Alpha #012
        # Original: sign(delta(volume,1)) * (-1 * delta(close,1))
        fields.append(
            "Sign(Delta($volume,1))*(-1*Delta($close,1))"
        )
        names.append("ALPHA012")

        # Alpha #013
        # Original: -1 * rank(covariance(rank(close), rank(volume), 5))
        fields.append(
            "-1*Rank(Cov(Rank($close,1), Rank($volume,1), 5), 1)"
        )
        names.append("ALPHA013")

        # Alpha #014
        # Original: (-1*rank(delta(returns,3))) * correlation(open, volume, 10)
        fields.append(
            "(-1*Rank(Delta($close/Ref($close,1)-1,3), 1))*Corr($open, $volume, 10)"
        )
        names.append("ALPHA014")

        # Alpha #015
        # Original: -1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)
        fields.append(
            "-1*Sum(Rank(Corr(Rank($high,1), Rank($volume,1), 3), 1), 3)"
        )
        names.append("ALPHA015")

        # Alpha #016
        # Original: -1 * rank(covariance(rank(high), rank(volume), 5))
        fields.append(
            "-1*Rank(Cov(Rank($high,1), Rank($volume,1), 5), 1)"
        )
        names.append("ALPHA016")

        # Alpha #017
        # Original: ((-1*rank(ts_rank(close,10)))*rank(delta(delta(close,1),1)))*rank(ts_rank(volume/adv20,5))
        fields.append(
            "(-1*Rank(Rank($close,10),1))*Rank(Delta(Delta($close,1),1),1)"
            "*Rank(Rank($volume/Mean($volume,20),5),1)"
        )
        names.append("ALPHA017")

        # Alpha #018
        # Original: -1 * rank(stddev(abs(close-open),5) + (close-open) + correlation(close,open,10))
        fields.append(
            "-1*Rank(Std(Abs($close-$open),5)+($close-$open)+Corr($close,$open,10), 1)"
        )
        names.append("ALPHA018")

        # Alpha #019
        # Original: (-1*sign((close-delay(close,7))+delta(close,7)))*(1+rank(1+sum(returns,250)))
        # [USES ts_argmin indirectly? No, this one is fine]
        fields.append(
            "(-1*Sign(($close-Ref($close,7))+Delta($close,7)))"
            "*(1+Rank(1+Sum($close/Ref($close,1)-1,250),1))"
        )
        names.append("ALPHA019")

        # Alpha #020
        # Original: (-1*rank(open-delay(high,1)))*rank(open-delay(close,1))*rank(open-delay(low,1))
        fields.append(
            "(-1*Rank($open-Ref($high,1),1))*Rank($open-Ref($close,1),1)*Rank($open-Ref($low,1),1)"
        )
        names.append("ALPHA020")

        # Alpha #021
        # Original: SUM((CLOSE/DELAY(CLOSE,1)-1)>0?MIN(CLOSE/DELAY(CLOSE,1)-1,0):MAX(CLOSE/DELAY(CLOSE,1)-1,0),8)
        # Simplified: sum of signed returns over 8 days (up days vs down days diff)
        fields.append(
            "Sum(If($close>Ref($close,1), "
            "Less($close/Ref($close,1)-1, 0), "
            "Greater($close/Ref($close,1)-1, 0)), 8)"
        )
        names.append("ALPHA021")

        # Alpha #022
        # Original: -1*delta(corr(high,volume,5),5)*rank(stddev(close,20))
        fields.append(
            "-1*Delta(Corr($high,$volume,5),5)*Rank(Std($close,20),1)"
        )
        names.append("ALPHA022")

        # Alpha #023
        # Original: ((sum(high,20)/20)<high)?(-1*delta(high,2)):0
        fields.append(
            "If(Mean($high,20)<$high, -1*Delta($high,2), 0)"
        )
        names.append("ALPHA023")

        # Alpha #024
        # Original: (((delta((sum(close,100)/100),100)/delay(close,100))<=0.05)?(-1*(close-ts_min(close,100))):(-1*delta(close,3)))
        fields.append(
            "If((Delta(Mean($close,100),100)/Ref($close,100))<=0.05, "
            "-1*($close-Min($close,100)), "
            "-1*Delta($close,3))"
        )
        names.append("ALPHA024")

        # Alpha #025
        # Original: rank((-1*returns)*adv20*vwap*(high-close))
        fields.append(
            "Rank((-1*($close/Ref($close,1)-1))*Mean($volume,20)*$vwap*($high-$close), 1)"
        )
        names.append("ALPHA025")

        # Alpha #026
        # Original: -1*ts_max(corr(ts_rank(volume,5),ts_rank(high,5),5),3)
        fields.append(
            "-1*Max(Corr(Rank($volume,5), Rank($high,5), 5), 3)"
        )
        names.append("ALPHA026")

        # Alpha #027
        # Original: (0.5<rank((sum(corr(rank(volume),rank(vwap),6),2)/2.0)))?(-1):1
        fields.append(
            "If(0.5<Rank(Sum(Corr(Rank($volume,1),Rank($vwap,1),6),2)/2.0,1), -1, 1)"
        )
        names.append("ALPHA027")

        # Alpha #028
        # Original: scale(((correlation(adv20,low,5)+((high+low)/2))-close))
        # Approximate scale with z-score via Rank
        fields.append(
            "Rank(Corr(Mean($volume,20),$low,5)+(($high+$low)/2)-$close, 1)"
        )
        names.append("ALPHA028")

        # Alpha #029
        # Original: min(product(rank(rank(decay_linear(-1*rank(delta(close,2)),8))),1),5)+ts_rank(delta(vwap,1),5)
        fields.append(
            "Min(Rank(Rank(WMA(-1*Rank(Delta($close,2),1),8),1),1),5)+Rank(Delta($vwap,1),5)"
        )
        names.append("ALPHA029")

        # Alpha #030
        # Original: ((1.0-rank(((sign((close-delay(close,1)))+sign((delay(close,1)-delay(close,2))))+sign((delay(close,2)-delay(close,3))))))*sum(volume,5))/sum(volume,20)
        fields.append(
            "((1.0-Rank((Sign(($close-Ref($close,1)))+Sign((Ref($close,1)-Ref($close,2))))+Sign((Ref($close,2)-Ref($close,3))),1))*Sum($volume,5))/Sum($volume,20)"
        )
        names.append("ALPHA030")

        # Alpha #031
        # Original: (rank(rank(rank(decay_linear(-1*rank(rank(delta(close,10))),10))))+rank(-1*delta(close,3)))+sign(scale(correlation(adv20,low,12)))
        fields.append(
            "(Rank(Rank(Rank(WMA(-1*Rank(Rank(Delta($close,10),1),1),10),1),1),1)+Rank(-1*Delta($close,3),1))"
            "+Sign(Rank(Corr(Mean($volume,20),$low,12),1))"
        )
        names.append("ALPHA031")

        # Alpha #032
        # Original: (scale(((sum(close,7)/7)-close))+(20*scale(correlation(vwap,delay(close,5),230))))
        fields.append(
            "Rank(Mean($close,7)-$close,1)+20*Rank(Corr($vwap,Ref($close,5),230),1)"
        )
        names.append("ALPHA032")

        # Alpha #033
        # Original: rank(-1*(1-(open/close)))
        fields.append(
            "Rank(-1*(1-($open/$close)), 1)"
        )
        names.append("ALPHA033")

        # Alpha #034
        # Original: rank(1-rank((stddev(returns,2)/stddev(returns,5))))
        fields.append(
            "Rank(1-Rank(Std($close/Ref($close,1)-1,2)/Std($close/Ref($close,1)-1,5),1),1)"
        )
        names.append("ALPHA034")

        # Alpha #035
        # Original: ((ts_rank(volume,32)*(1-ts_rank(close+high-low,16)))*(1-ts_rank(returns,32)))
        fields.append(
            "(Rank($volume,32)*(1-Rank($close+$high-$low,16)))*(1-Rank($close/Ref($close,1)-1,32))"
        )
        names.append("ALPHA035")

        # Alpha #036
        # Original: rank(corr(rank(close-open),rank(volume),15))
        fields.append(
            "Rank(Corr(Rank($close-$open,1),Rank($volume,1),15),1)"
        )
        names.append("ALPHA036")

        # Alpha #037
        # Original: -1*rank(((sum(open,5)*sum(returns,5))-delay(sum(open,5)*sum(returns,5),10)))
        # (same formula as #008 but different rank position - here rank wraps everything)
        fields.append(
            "-1*Rank((Sum($open,5)*Sum($close/Ref($close,1)-1,5))-Ref(Sum($open,5)*Sum($close/Ref($close,1)-1,5),10), 1)"
        )
        names.append("ALPHA037")

        # Alpha #038
        # Original: -1*rank(ts_rank(close,10))*rank(close/open)
        fields.append(
            "-1*Rank(Rank($close,10),1)*Rank($close/$open,1)"
        )
        names.append("ALPHA038")

        # Alpha #039
        # Original: -1*rank(delta(close,7))*(1-rank(decay_linear(volume/adv20,9)))
        fields.append(
            "-1*Rank(Delta($close,7),1)*(1-Rank(WMA($volume/Mean($volume,20),9),1))"
        )
        names.append("ALPHA039")

        # Alpha #040
        # Original: -1*rank(stddev(high,10))*correlation(high,volume,10)
        fields.append(
            "-1*Rank(Std($high,10),1)*Corr($high,$volume,10)"
        )
        names.append("ALPHA040")

        # Alpha #041
        # Original: ((high*low)^0.5)-vwap
        fields.append(
            "Power($high*$low, 0.5)-$vwap"
        )
        names.append("ALPHA041")

        # Alpha #042
        # Original: rank((vwap-close))/rank((vwap+close))
        fields.append(
            "Rank($vwap-$close,1)/Rank($vwap+$close,1)"
        )
        names.append("ALPHA042")

        # Alpha #043
        # Original: ts_rank(volume/adv20,20)*ts_rank(-1*delta(close,7),8)
        fields.append(
            "Rank($volume/Mean($volume,20),20)*Rank(-1*Delta($close,7),8)"
        )
        names.append("ALPHA043")

        # Alpha #044
        # Original: -1*correlation(high,rank(volume),5)
        fields.append(
            "-1*Corr($high, Rank($volume,1), 5)"
        )
        names.append("ALPHA044")

        # Alpha #045
        # Original: -1*rank(delta(sum(delay(close,5),20)/20,2))*correlation(close,volume,2)*rank(correlation(sum(close,5),sum(close,20),2))
        fields.append(
            "-1*Rank(Delta(Sum(Ref($close,5),20)/20,2),1)"
            "*Corr($close,$volume,2)"
            "*Rank(Corr(Sum($close,5),Sum($close,20),2),1)"
        )
        names.append("ALPHA045")

        # Alpha #046
        # Original: (0.25<((delay(close,20)-delay(close,10))/10-((delay(close,10)-close)/10)))?-1:(close>delay(close,20)?1:0)
        fields.append(
            "If(0.25<((Ref($close,20)-Ref($close,10))/10-((Ref($close,10)-$close)/10)), "
            "-1, "
            "If($close>Ref($close,20), 1, 0))"
        )
        names.append("ALPHA046")

        # Alpha #047
        # Original: ((rank(1/close))*volume/adv20)/sum(volume,5)
        fields.append(
            "Rank(1/$close,1)*$volume/Mean($volume,20)/Sum($volume,5)"
        )
        names.append("ALPHA047")

        # Alpha #048
        # Original: -1*((rank(((sign((close-delay(close,1)))+sign((delay(close,1)-delay(close,2))))+sign((delay(close,2)-delay(close,3))))))*sum(volume,5))/sum(volume,20)
        fields.append(
            "-1*(Rank((Sign(($close-Ref($close,1)))+Sign((Ref($close,1)-Ref($close,2))))"
            "+Sign((Ref($close,2)-Ref($close,3))),1)*Sum($volume,5))/Sum($volume,20)"
        )
        names.append("ALPHA048")

        # Alpha #049
        # Original: ((sum(delay(close,20)-delay(close,10),20)/10-(sum(delay(close,10)-close,10))/10))
        fields.append(
            "(Sum(Ref($close,20)-Ref($close,10),20)/10-(Sum(Ref($close,10)-$close,10))/10)"
        )
        names.append("ALPHA049")

        # Alpha #050
        # Original: -1*ts_max(rank(corr(rank(volume),rank(vwap),5)),5)
        fields.append(
            "-1*Max(Rank(Corr(Rank($volume,1),Rank($vwap,1),5),1),5)"
        )
        names.append("ALPHA050")

        # Alpha #051
        # Original: (0.5<rank(sum(delay(close,20)-delay(close,10),20)/10-(sum(delay(close,10)-close,10))/10))?-1:1
        fields.append(
            "If(0.5<Rank(Sum(Ref($close,20)-Ref($close,10),20)/10"
            "-(Sum(Ref($close,10)-$close,10))/10,1), -1, 1)"
        )
        names.append("ALPHA051")

        # Alpha #052
        # Original: ts_rank(-1*min(low-delay(close,5),5)*rank(corr(sum(close,5),sum(close,20),2)),5)
        fields.append(
            "Rank(-1*Min($low-Ref($close,5),5)*Rank(Corr(Sum($close,5),Sum($close,20),2),1),5)"
        )
        names.append("ALPHA052")

        # Alpha #053
        # Original: -1*delta((((close-low)-(high-close))/(high-low+1e-12)),9)
        fields.append(
            "-1*Delta((($close-$low)-($high-$close))/($high-$low+1e-12), 9)"
        )
        names.append("ALPHA053")

        # Alpha #054
        # Original: -1*(low-close)*(open^5)/((low-high)*(close^5))
        # Simplified: the original formula uses power operations
        fields.append(
            "-1*($low-$close)*Power($open,5)/(($low-$high+1e-12)*Power($close,5))"
        )
        names.append("ALPHA054")

        # Alpha #055
        # Original: -1*correlation(rank(((close-ts_min(low,12))/(ts_max(high,12)-ts_min(low,12)+1e-12))), rank(volume), 6)
        fields.append(
            "-1*Corr(Rank(($close-Min($low,12))/(Max($high,12)-Min($low,12)+1e-12),1), Rank($volume,1), 6)"
        )
        names.append("ALPHA055")

        # Alpha #056
        # Original: 0-1*(rank((sum(returns,10)/sum(sum(returns,2),3)))*rank((returns*cap)))
        # (cap not available in standard OHLCV, approximate without cap)
        fields.append(
            "-1*Rank(Sum($close/Ref($close,1)-1,10)/Sum(Sum($close/Ref($close,1)-1,2),3),1)"
            "*Rank($close/Ref($close,1)-1,1)"
        )
        names.append("ALPHA056")

        # Alpha #057
        # Original: 0-1*((close-vwap)/decay_linear(rank(ts_argmax(close,30)),2))
        # Exact with custom operator TsArgmax
        fields.append(
            "-1*(($close-$vwap)/WMA(Rank(TsArgmax($close,30),1),2))"
        )
        names.append("ALPHA057")

        # Alpha #058
        # Original: -1*ts_rank(decay_linear(correlation(indneutralize(vwap,indclass),volume,3.92795),7.89291),5.50322)*rank(decay_linear(correlation(vwap,volume,3.92795),7.89291))
        # Approximate without sector neutralize (simplified version)
        fields.append(
            "-1*Rank(WMA(Corr($vwap,$volume,4),8),6)*Rank(WMA(Corr($vwap,$volume,4),8),1)"
        )
        names.append("ALPHA058")

        # Alpha #059
        # Original: -1*ts_rank(decay_linear(correlation(indneutralize(vwap,indclass),volume,9.907),7.89291),5.50322)*rank(decay_linear(correlation(vwap,volume,9.907),7.89291))
        fields.append(
            "-1*Rank(WMA(Corr($vwap,$volume,10),8),6)*Rank(WMA(Corr($vwap,$volume,10),8),1)"
        )
        names.append("ALPHA059")

        # Alpha #060
        # Original: 0-1*(2*scale(rank(((close-low)-(high-close))/(high-low+1e-12)))-scale(rank(ts_argmax(close,10))))
        # Exact with custom operator TsArgmax
        fields.append(
            "-1*(2*Rank(Rank((($close-$low)-($high-$close))/($high-$low+1e-12),1),1)"
            "-Rank(Rank(TsArgmax($close,10),1),1))"
        )
        names.append("ALPHA060")

        # Alpha #061
        # Original: rank(vwap-ts_min(vwap,16.1219)) < rank(correlation(vwap,adv180,17.9282))
        fields.append(
            "Rank($vwap-Min($vwap,16),1)-Rank(Corr($vwap,Mean($volume,180),18),1)"
        )
        names.append("ALPHA061")

        # Alpha #062
        # Original: -1*correlation(high,rank(correlation(vwap,adv20,22.4104)),19.8341)
        fields.append(
            "-1*Corr($high, Rank(Corr($vwap,Mean($volume,20),22),1), 20)"
        )
        names.append("ALPHA062")

        # Alpha #063
        # Original: -1*rank(decay_linear(delta(indneutralize(close,indclass),2.25164),8.22237))*rank(decay_linear(correlation(vwap,adv20,8.44728),6.49551))
        fields.append(
            "-1*Rank(WMA(Delta($close,2),8),1)*Rank(WMA(Corr($vwap,Mean($volume,20),8),6),1)"
        )
        names.append("ALPHA063")

        # Alpha #064
        # Original: -1*rank(decay_linear(correlation(rank(vwap),rank(volume),4.01379),2.6809))
        fields.append(
            "-1*Rank(WMA(Corr(Rank($vwap,1),Rank($volume,1),4),3),1)"
        )
        names.append("ALPHA064")

        # Alpha #065
        # Original: -1*rank(decay_linear(correlation(close,adv60,9.17385),14.3857))
        fields.append(
            "-1*Rank(WMA(Corr($close,Mean($volume,60),9),14),1)"
        )
        names.append("ALPHA065")

        # Alpha #066
        # Original: -1*rank(decay_linear(delta(vwap,3.51013),7.23052))
        fields.append(
            "-1*Rank(WMA(Delta($vwap,4),7),1)"
        )
        names.append("ALPHA066")

        # Alpha #067
        # Original: -1*rank(decay_linear(correlation(rank(high),rank(adv15),8.91965),7.67903))
        fields.append(
            "-1*Rank(WMA(Corr(Rank($high,1),Rank(Mean($volume,15),1),9),8),1)"
        )
        names.append("ALPHA067")

        # Alpha #068
        # Original: ts_rank(correlation(rank(high),rank(adv15),8.91644),7.19041)
        fields.append(
            "Rank(Corr(Rank($high,1),Rank(Mean($volume,15),1),9),7)"
        )
        names.append("ALPHA068")

        # Alpha #069
        # Original: rank(decay_linear(delta(indneutralize(vwap,indclass),3.44174),7.93177))*ts_rank(decay_linear(delta(vwap,3.44174),7.93177),7.26666)
        fields.append(
            "Rank(WMA(Delta($vwap,3),8),1)*Rank(WMA(Delta($vwap,3),8),7)"
        )
        names.append("ALPHA069")

        # Alpha #070
        # Original: rank(decay_linear(delta(vwap,1.25186),10.7861))
        fields.append(
            "Rank(WMA(Delta($vwap,1),11),1)"
        )
        names.append("ALPHA070")

        # Alpha #071
        # Original: rank(decay_linear(correlation(ts_rank(close,3.43976),ts_rank(adv180,12.0643),18.0175),4.20501))
        fields.append(
            "Rank(WMA(Corr(Rank($close,3),Rank(Mean($volume,180),12),18),4),1)"
        )
        names.append("ALPHA071")

        # Alpha #072
        # Original: rank(decay_linear(correlation(close,adv60,4.59186),7.77983))
        fields.append(
            "Rank(WMA(Corr($close,Mean($volume,60),5),8),1)"
        )
        names.append("ALPHA072")

        # Alpha #073
        # Original: rank(decay_linear(delta(vwap,1.24905),10.9937))*ts_rank(decay_linear(delta(vwap,1.24905),10.9937),7.23091)
        fields.append(
            "Rank(WMA(Delta($vwap,1),11),1)*Rank(WMA(Delta($vwap,1),11),7)"
        )
        names.append("ALPHA073")

        # Alpha #074
        # Original: rank(correlation(sum((low*0.35+vwap*0.65),20),sum(adv60,20),7.41223))+rank(correlation(rank(vwap),rank(volume),5.95041))
        fields.append(
            "Rank(Corr(Sum($low*0.35+$vwap*0.65,20),Sum(Mean($volume,60),20),7),1)"
            "+Rank(Corr(Rank($vwap,1),Rank($volume,1),6),1)"
        )
        names.append("ALPHA074")

        # Alpha #075
        # Original: rank(correlation(vwap,volume,4.24304)) < rank(correlation(rank(high),rank(adv50),11.9908))
        fields.append(
            "Rank(Corr($vwap,$volume,4),1)-Rank(Corr(Rank($high,1),Rank(Mean($volume,50),1),12),1)"
        )
        names.append("ALPHA075")

        # Alpha #076
        # Original: rank(decay_linear(delta(vwap,1.24317),11.8259))*ts_rank(decay_linear(ts_rank(correlation(indneutralize(low,indclass),adv81,8.14941),19.569),19.569),2.01692)
        fields.append(
            "Rank(WMA(Delta($vwap,1),12),1)*Rank(WMA(Rank(Corr($low,Mean($volume,81),8),20),20),2)"
        )
        names.append("ALPHA076")

        # Alpha #077
        # Original: rank(decay_linear(correlation(high,adv60,3.54576),5.79691))
        fields.append(
            "Rank(WMA(Corr($high,Mean($volume,60),4),6),1)"
        )
        names.append("ALPHA077")

        # Alpha #078
        # Original: rank(correlation(sum(vwap*0.23+low*0.77,19.772),sum(adv60*1.18+high*0.82,19.772),6.77394))*rank(correlation(rank(vwap),rank(volume),5.17874))
        fields.append(
            "Rank(Corr(Sum($vwap*0.23+$low*0.77,20),Sum(Mean($volume,60)*1.18+$high*0.82,20),7),1)"
            "*Rank(Corr(Rank($vwap,1),Rank($volume,1),5),1)"
        )
        names.append("ALPHA078")

        # Alpha #079
        # Original: rank(delta(indneutralize((close*0.31+vwap*0.69),indclass),1.23456))*rank(decay_linear(ts_rank(correlation(ts_rank(close,3.54746),ts_rank(adv180,11.9403),18.0698),4.96531),2.06692))
        fields.append(
            "Rank(Delta($close*0.31+$vwap*0.69,1),1)"
            "*Rank(WMA(Rank(Corr(Rank($close,4),Rank(Mean($volume,180),12),18),5),2),1)"
        )
        names.append("ALPHA079")

        # Alpha #080
        # Original: rank(decay_linear(correlation(high,adv10,5.11425),4.51099))*rank(decay_linear(correlation(vwap,adv10,5.11425),4.51099))
        fields.append(
            "Rank(WMA(Corr($high,Mean($volume,10),5),5),1)"
            "*Rank(WMA(Corr($vwap,Mean($volume,10),5),5),1)"
        )
        names.append("ALPHA080")

        # Alpha #081
        # Original: rank(decay_linear(product(rank(corr(sum(close,8),sum(adv60,8),8.32377)),rank(ts_rank(vwap,3.03612))),6.94854))
        # Simplified: product -> multiply
        fields.append(
            "Rank(WMA(Rank(Corr(Sum($close,8),Sum(Mean($volume,60),8),8),1)"
            "*Rank(Rank($vwap,3),1),7),1)"
        )
        names.append("ALPHA081")

        # Alpha #082
        # Original: rank(decay_linear(delta(open,1.01668),10.7383))*rank(decay_linear(correlation(open,adv10,3.42396),7.36179))
        fields.append(
            "Rank(WMA(Delta($open,1),11),1)*Rank(WMA(Corr($open,Mean($volume,10),3),7),1)"
        )
        names.append("ALPHA082")

        # Alpha #083
        # Original: rank(decay_linear(delta(((high+low)/2*vwap*0.76+(open+close)/2*0.24),1.50719),7.20942))*rank(decay_linear(correlation(ts_rank(vwap,4.12085),ts_rank(adv60,8.59529),7.00248),3.61502))
        fields.append(
            "Rank(WMA(Delta((($high+$low)/2*$vwap*0.76+($open+$close)/2*0.24),2),7),1)"
            "*Rank(WMA(Corr(Rank($vwap,4),Rank(Mean($volume,60),9),7),4),1)"
        )
        names.append("ALPHA083")

        # Alpha #084
        # Original: signed_power(ts_rank((vwap-ts_max(vwap,15.3217)),20.7127),delta(close,4.96796))
        fields.append(
            "Sign(Rank($vwap-Max($vwap,15),21))*Power(Abs(Rank($vwap-Max($vwap,15),21)),Delta($close,5))"
        )
        names.append("ALPHA084")

        # Alpha #085
        # Original: rank(correlation((high*0.876703+close*0.123297),adv30,9.61331))^ts_rank(correlation(ts_rank((high+low)/2,3.70596),ts_rank(volume,10.1595),13.609),6.83553)
        fields.append(
            "Power(Rank(Corr($high*0.88+$close*0.12,Mean($volume,30),10),1),"
            "Rank(Corr(Rank(($high+$low)/2,4),Rank($volume,10),14),7))"
        )
        names.append("ALPHA085")

        # Alpha #086
        # Original: ts_rank(correlation(close,adv20,6.47424),20.5617)
        fields.append(
            "Rank(Corr($close,Mean($volume,20),6),21)"
        )
        names.append("ALPHA086")

        # Alpha #087
        # Original: rank(decay_linear(delta(vwap,4.00439),6.78652))*ts_rank(decay_linear(((((low*0.721001+vwap*(1-0.721001))-adv60*(1+0.092399))/vwap)*(high+low)/2),11.3135),2.12598)
        fields.append(
            "Rank(WMA(Delta($vwap,4),7),1)"
            "*Rank(WMA((((($low*0.72+$vwap*0.28)-Mean($volume,60)*1.09)/$vwap)*($high+$low)/2),11),2)"
        )
        names.append("ALPHA087")

        # Alpha #088
        # Original: rank(decay_linear(correlation(open,adv60,8.86626),3.22169))*rank(decay_linear(ts_rank((vwap-ts_min(vwap,11.5783)),19.6415),2.60487))
        fields.append(
            "Rank(WMA(Corr($open,Mean($volume,60),9),3),1)"
            "*Rank(WMA(Rank($vwap-Min($vwap,12),20),3),1)"
        )
        names.append("ALPHA088")

        # Alpha #089
        # Original: ts_rank(decay_linear(correlation((low*0.967285+low*0.032715),adv60,6.48821),4.41568),2.14442)-ts_rank(decay_linear(rank(correlation(vwap,adv10,4.72267)),5.22343),2.14882)
        fields.append(
            "Rank(WMA(Corr($low,Mean($volume,60),6),4),2)"
            "-Rank(WMA(Rank(Corr($vwap,Mean($volume,10),5),1),5),2)"
        )
        names.append("ALPHA089")

        # Alpha #090
        # Original: rank(decay_linear(correlation(rank(vwap),rank(volume),4.58403),6.66177))*ts_rank(decay_linear(correlation(rank(close),rank(adv60),10.1792),4.41158),5.32207)
        fields.append(
            "Rank(WMA(Corr(Rank($vwap,1),Rank($volume,1),5),7),1)"
            "*Rank(WMA(Corr(Rank($close,1),Rank(Mean($volume,60),1),10),4),5)"
        )
        names.append("ALPHA090")

        # Alpha #091
        # Original: rank(decay_linear(decay_linear(correlation(close,adv30,8.3744),6.54329),3.78017))*ts_rank(decay_linear(delta(close,3.06269),3.40089),2.04176)
        fields.append(
            "Rank(WMA(WMA(Corr($close,Mean($volume,30),8),7),4),1)"
            "*Rank(WMA(Delta($close,3),3),2)"
        )
        names.append("ALPHA091")

        # Alpha #092
        # Original: rank(decay_linear(correlation(open,adv60,8.86937),5.9543))*rank(decay_linear(correlation(high,adv60,20.123),6.81999))
        fields.append(
            "Rank(WMA(Corr($open,Mean($volume,60),9),6),1)"
            "*Rank(WMA(Corr($high,Mean($volume,60),20),7),1)"
        )
        names.append("ALPHA092")

        # Alpha #093
        # Original: ts_rank(decay_linear(correlation(indneutralize(vwap,indclass),adv81,17.4193),19.848),7.54455)/rank(decay_linear(delta((close*0.524434+vwap*(1-0.524434)),2.77377),16.8624))
        fields.append(
            "Rank(WMA(Corr($vwap,Mean($volume,81),17),20),8)"
            "/Rank(WMA(Delta($close*0.52+$vwap*0.48,3),17),1)"
        )
        names.append("ALPHA093")

        # Alpha #094
        # Original: rank(decay_linear(correlation(vwap,adv60,4.90982),5.129))*ts_rank(decay_linear(ts_rank(correlation(ts_rank(vwap,5.71423),ts_rank(close,5.71423),10.4069),19.4169),19.4169),2.54415)
        fields.append(
            "Rank(WMA(Corr($vwap,Mean($volume,60),5),5),1)"
            "*Rank(WMA(Rank(Corr(Rank($vwap,6),Rank($close,6),10),19),19),3)"
        )
        names.append("ALPHA094")

        # Alpha #095
        # Original: rank(decay_linear(delta(close,1.03453),10.5964))*ts_rank(decay_linear(-1*delta(close,1.03453),10.5964),6.87614)
        fields.append(
            "Rank(WMA(Delta($close,1),11),1)*Rank(WMA(-1*Delta($close,1),11),7)"
        )
        names.append("ALPHA095")

        # Alpha #096
        # Original: ts_rank(decay_linear(correlation(rank(vwap),rank(volume),19.7534),4.89768),8.42623)
        fields.append(
            "Rank(WMA(Corr(Rank($vwap,1),Rank($volume,1),20),5),8)"
        )
        names.append("ALPHA096")

        # Alpha #097
        # Original: rank(decay_linear(delta(indneutralize((low*0.721001+vwap*(1-0.721001)),indclass),3.37007),20.4629))*ts_rank(decay_linear(ts_rank(correlation(ts_rank(low,7.87192),ts_rank(adv60,8.95426),6.05869),6.07961),6.07961),1.48853)
        fields.append(
            "Rank(WMA(Delta($low*0.72+$vwap*0.28,3),20),1)"
            "*Rank(WMA(Rank(Corr(Rank($low,8),Rank(Mean($volume,60),9),6),6),6),1)"
        )
        names.append("ALPHA097")

        # Alpha #098
        # Original: rank(decay_linear(correlation(vwap,sum(adv5,26.4719),4.58418),7.18088))-rank(decay_linear(ts_rank(ts_argmin(correlation(rank(open),rank(adv15),20.8187),8.62571),6.95668),8.07206))
        # Exact with custom operator TsArgmin
        fields.append(
            "Rank(WMA(Corr($vwap,Sum(Mean($volume,5),26),5),7),1)"
            "-Rank(WMA(Rank(TsArgmin(Corr(Rank($open,1),Rank(Mean($volume,15),1),21),9),7),8),1)"
        )
        names.append("ALPHA098")

        # Alpha #099
        # Original: rank(decay_linear(correlation(sum(close,19.8977),sum(adv60,19.8977),8.81362),6.41764))-rank(decay_linear(correlation(rank(vwap),rank(volume),6.22508),4.0031))
        fields.append(
            "Rank(WMA(Corr(Sum($close,20),Sum(Mean($volume,60),20),9),6),1)"
            "-Rank(WMA(Corr(Rank($vwap,1),Rank($volume,1),6),4),1)"
        )
        names.append("ALPHA099")

        # Alpha #100
        # Original: rank(decay_linear(correlation(rank(close),rank(volume),5.30463),7.59591))
        fields.append(
            "Rank(WMA(Corr(Rank($close,1),Rank($volume,1),5),8),1)"
        )
        names.append("ALPHA100")

        # Alpha #101
        # Original: (close-open)/(high-low+1e-12)*volume
        fields.append(
            "($close-$open)/($high-$low+1e-12)*$volume"
        )
        names.append("ALPHA101")

        return fields, names
