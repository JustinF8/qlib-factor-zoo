# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# GTJA191 Data Loader - 191 Alphas from GTJA Research Report (2014)
# Converted from Vibe-Trading alpha zoo for use with Qlib expression engine.
#
# Reference: 国泰君安 191 alpha 研报 (2014)
#
# CUSTOM OPERATORS REQUIRED (auto-registered via handler.py):
#   - SMA(X, N, M)  : Simple Moving Average (Chinese finance convention)
#   - TsArgmax(X, N) : Rolling argmax value extraction
#   - TsArgmin(X, N) : Rolling argmin value extraction
#   - Amount()       : 成交额 accessor (uses $vwap*$volume as proxy)
#
# Qlib built-in operators used:
#   - EMA(X, N)      : Exponential Moving Average (span=N)
#   - WMA(X, N)      : Weighted Moving Average (linear weights)
#   - Mean(X, N)     : Simple Moving Average (= SMA with M=1)

from qlib.data.dataset.loader import QlibDataLoader


class GTJA191DL(QlibDataLoader):
    """Data loader for GTJA 191 Alphas."""

    def __init__(self, config=None, **kwargs):
        _config = {"feature": self.get_feature_config()}
        if config is not None:
            _config.update(config)
        super().__init__(config=_config, **kwargs)

    @staticmethod
    def get_feature_config():
        fields = []
        names = []

        # 001: -1*CORR(RANK(DELTA(LOG(VOLUME),1)), RANK((CLOSE-OPEN)/OPEN), 6)
        fields.append(
            "-1*Corr(Rank(Delta(Log($volume+1),1),1), Rank(($close-$open)/$open,1), 6)"
        )
        names.append("GTJA001")

        # 002: -1*DELTA(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW), 1)
        fields.append(
            "-1*Delta((($close-$low)-($high-$close))/($high-$low+1e-12), 1)"
        )
        names.append("GTJA002")

        # 003: SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
        # Wilder's directional movement sum
        fields.append(
            "Sum(If($close>Ref($close,1), "
            "$close-Less($low,Ref($close,1)), "
            "If($close<Ref($close,1), "
            "$close-Greater($high,Ref($close,1)), 0)), 6)"
        )
        names.append("GTJA003")

        # 004: complex ternary with volume condition
        fields.append(
            "If(Mean($close,8)+Std($close,8)<Mean($close,2), -1, "
            "If(Mean($close,2)<Mean($close,8)-Std($close,8), 1, "
            "If($volume/Mean($volume,20)>1, 1, -1)))"
        )
        names.append("GTJA004")

        # 005: -1*TSMAX(CORR(TSRANK(VOLUME,5), TSRANK(HIGH,5), 5), 3)
        fields.append(
            "-1*Max(Corr(Rank($volume,5), Rank($high,5), 5), 3)"
        )
        names.append("GTJA005")

        # 006: -1*RANK(SIGN(DELTA((OPEN*0.85+HIGH*0.15), 4)))
        fields.append(
            "-1*Rank(Sign(Delta($open*0.85+$high*0.15, 4)), 1)"
        )
        names.append("GTJA006")

        # 007: (RANK(MAX(VWAP-CLOSE,3))+RANK(MIN(VWAP-CLOSE,3)))*RANK(DELTA(VOLUME,3))
        fields.append(
            "(Rank(Max($vwap-$close,3),1)+Rank(Min($vwap-$close,3),1))*Rank(Delta($volume,3),1)"
        )
        names.append("GTJA007")

        # 008: -1*RANK(DELTA(((HIGH+LOW)/2*0.2+VWAP*0.8), 4))
        fields.append(
            "-1*Rank(Delta((($high+$low)/2)*0.2+$vwap*0.8, 4), 1)"
        )
        names.append("GTJA008")

        # 009: SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME, 7, 2)
        fields.append(
            "SMA((($high+$low)/2-(Ref($high,1)+Ref($low,1))/2)*($high-$low)/($volume+1e-12), 7, 2)"
        )
        names.append("GTJA009")

        # 010: RANK(MAX(((RET<0)?STD(RET,20):CLOSE)^2, 5))
        fields.append(
            "Rank(Max(Power(If($close/Ref($close,1)-1<0, "
            "Std($close/Ref($close,1)-1,20), $close), 2), 5), 1)"
        )
        names.append("GTJA010")

        # 011: SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME, 6)
        fields.append(
            "Sum((($close-$low)-($high-$close))/($high-$low+1e-12)*$volume, 6)"
        )
        names.append("GTJA011")

        # 012: RANK(OPEN-SUM(VWAP,10)/10)*(-1*RANK(ABS(CLOSE-VWAP)))
        fields.append(
            "Rank($open-Sum($vwap,10)/10,1)*(-1*Rank(Abs($close-$vwap),1))"
        )
        names.append("GTJA012")

        # 013: (HIGH*LOW)^0.5 - VWAP
        fields.append(
            "Power($high*$low, 0.5)-$vwap"
        )
        names.append("GTJA013")

        # 014: CLOSE-DELAY(CLOSE,5)
        fields.append(
            "Delta($close, 5)"
        )
        names.append("GTJA014")

        # 015: OPEN/DELAY(CLOSE,1) - 1
        fields.append(
            "$open/Ref($close,1)-1"
        )
        names.append("GTJA015")

        # 016: -1*TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5)
        fields.append(
            "-1*Max(Rank(Corr(Rank($volume,1), Rank($vwap,1), 5), 1), 5)"
        )
        names.append("GTJA016")

        # 017: RANK(VWAP-MAX(VWAP,15))^DELTA(CLOSE,5)
        # Approximate with Sign*Power(Abs)
        fields.append(
            "Sign(Rank($vwap-Max($vwap,15),1))*Power(Abs(Rank($vwap-Max($vwap,15),1)), Abs(Delta($close,5)))"
        )
        names.append("GTJA017")

        # 018: CLOSE/DELAY(CLOSE,5)
        fields.append(
            "$close/Ref($close, 5)"
        )
        names.append("GTJA018")

        # 019: Conditional return based on direction
        fields.append(
            "If($close<Ref($close,5), ($close-Ref($close,5))/Ref($close,5), "
            "If($close>Ref($close,5), ($close-Ref($close,5))/$close, 0))"
        )
        names.append("GTJA019")

        # 020: (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100
        fields.append(
            "($close-Ref($close,6))/Ref($close,6)*100"
        )
        names.append("GTJA020")

        # 021: REGBETA(MEAN(CLOSE,6), SEQUENCE(6)) - slope of MA6 over 6 bars
        fields.append(
            "Slope(Mean($close,6), 6)/$close"
        )
        names.append("GTJA021")

        # 022: SMA((CLOSE-MA6)/MA6 - DELAY((CLOSE-MA6)/MA6, 3), 12, 1)
        fields.append(
            "SMA(($close-Mean($close,6))/Mean($close,6)-Ref(($close-Mean($close,6))/Mean($close,6),3), 12, 1)"
        )
        names.append("GTJA022")

        # 023: SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1) / (SMA(up)+SMA(down)) * 100
        fields.append(
            "SMA(If($close>Ref($close,1), Std($close,20), 0), 20, 1)"
            "/(SMA(If($close>Ref($close,1), Std($close,20), 0), 20, 1)"
            "+SMA(If($close<=Ref($close,1), Std($close,20), 0), 20, 1)+1e-12)*100"
        )
        names.append("GTJA023")

        # 024: SMA(CLOSE-DELAY(CLOSE,5), 5, 1)
        fields.append(
            "SMA(Delta($close,5), 5, 1)"
        )
        names.append("GTJA024")

        # 025: (-1*RANK(DELTA(CLOSE,7)*(1-RANK(DECAYLINEAR(VOL/ADV20,9)))))*(1+RANK(SUM(RET,250)))
        fields.append(
            "(-1*Rank(Delta($close,7)*(1-Rank(WMA($volume/Mean($volume,20),9),1)),1))"
            "*(1+Rank(Sum($close/Ref($close,1)-1,60),1))"
        )
        names.append("GTJA025")

        # 026: (SUM(CLOSE,7)/7-CLOSE)+CORR(VWAP,DELAY(CLOSE,5),230)
        fields.append(
            "(Mean($close,7)-$close)+Corr($vwap, Ref($close,5), 30)"
        )
        names.append("GTJA026")

        # 027: WMA(returns*100 sum, 12)
        fields.append(
            "WMA(($close-Ref($close,3))/Ref($close,3)*100+($close-Ref($close,6))/Ref($close,6)*100, 12)"
        )
        names.append("GTJA027")

        # 028: 3*SMA(RSV,3,1)-2*SMA(SMA(RSV,3,1),3,1) - KDJ-like
        rsv = "($close-Min($low,9))/(Max($high,9)-Min($low,9)+1e-12)*100"
        fields.append(
            f"3*SMA({rsv},3,1)-2*SMA(SMA({rsv},3,1),3,1)"
        )
        names.append("GTJA028")

        # 029: (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME
        fields.append(
            "($close-Ref($close,6))/Ref($close,6)*$volume"
        )
        names.append("GTJA029")

        # 030: WMA((CLOSE/DELAY(CLOSE,1)-1)*100, 2)
        fields.append(
            "WMA(($close/Ref($close,1)-1)*100, 2)"
        )
        names.append("GTJA030")

        # 031: (CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100
        fields.append(
            "($close-Mean($close,12))/Mean($close,12)*100"
        )
        names.append("GTJA031")

        # 032: -1*CORR(HIGH, RANK(VOLUME), 5)
        fields.append(
            "-1*Corr($high, Rank($volume,1), 5)"
        )
        names.append("GTJA032")

        # 033: -1*TSMIN(LOW,5)+DELAY(TSMIN(LOW,5),5)
        fields.append(
            "-1*Min($low,5)+Ref(Min($low,5),5)"
        )
        names.append("GTJA033")

        # 034: MEAN(CLOSE,12)/CLOSE
        fields.append(
            "Mean($close,12)/$close"
        )
        names.append("GTJA034")

        # 035: (OPEN-DELAY(CLOSE,1))/DELAY(CLOSE,1)*VOLUME
        fields.append(
            "($open-Ref($close,1))/Ref($close,1)*$volume"
        )
        names.append("GTJA035")

        # 036: RANK(CORR(CLOSE,VOLUME,15))*RANK(DELTA(CLOSE,5))
        fields.append(
            "Rank(Corr($close,$volume,15),1)*Rank(Delta($close,5),1)"
        )
        names.append("GTJA036")

        # 037: -1*RANK(DELTA(OPEN,1))
        fields.append(
            "-1*Rank(Delta($open,1),1)"
        )
        names.append("GTJA037")

        # 038: (SUM(HIGH,20)/20<HIGH)?-1*DELTA(HIGH,2):0
        fields.append(
            "If(Mean($high,20)<$high, -1*Delta($high,2), 0)"
        )
        names.append("GTJA038")

        # 039: DELTA(CLOSE,7)*(1-RANK(DECAYLINEAR(VOLUME/ADV20,9)))*(-1)
        fields.append(
            "Delta($close,7)*(1-Rank(WMA($volume/Mean($volume,20),9),1))*(-1)"
        )
        names.append("GTJA039")

        # 040: SUM(CLOSE>DELAY(CLOSE,1)?VOLUME:0,26)/SUM(CLOSE<=DELAY(CLOSE,1)?VOLUME:0,26)
        fields.append(
            "Sum(If($close>Ref($close,1), $volume, 0), 26)"
            "/(Sum(If($close<=Ref($close,1), $volume, 0), 26)+1e-12)"
        )
        names.append("GTJA040")

        # 041: RANK(MAX(DELTA(VWAP,3),5))*(-1)
        fields.append(
            "-1*Rank(Max(Delta($vwap,3),5),1)"
        )
        names.append("GTJA041")

        # 042: -1*RANK(STD(HIGH,10))*CORR(HIGH,VOLUME,10)
        fields.append(
            "-1*Rank(Std($high,10),1)*Corr($high,$volume,10)"
        )
        names.append("GTJA042")

        # 043: SUM(CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0),6)
        fields.append(
            "Sum(If($close>Ref($close,1), $volume, If($close<Ref($close,1), -$volume, 0)), 6)"
        )
        names.append("GTJA043")

        # 044: TSRANK(DECAYLINEAR(CORR(LOW,MEAN(VOLUME,10),7.5),6),4)
        fields.append(
            "Rank(WMA(Corr($low,Mean($volume,10),8),6),4)"
        )
        names.append("GTJA044")

        # 045: -1*RANK(DELTA(SUM(CLOSE,5)/5,2))*RANK(CORR(CLOSE,OPEN,5))
        fields.append(
            "-1*Rank(Delta(Mean($close,5),2),1)*Rank(Corr($close,$open,5),1)"
        )
        names.append("GTJA045")

        # 046: (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)
        fields.append(
            "(Mean($close,3)+Mean($close,6)+Mean($close,12)+Mean($close,24))/(4*$close)"
        )
        names.append("GTJA046")

        # 047: SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100, 9, 1)
        fields.append(
            "SMA((Max($high,6)-$close)/(Max($high,6)-Min($low,6)+1e-12)*100, 9, 1)"
        )
        names.append("GTJA047")

        # 048: -1*RANK(SIGN(CLOSE-DELAY(CLOSE,1))+SIGN(DELAY(CLOSE,1)-DELAY(CLOSE,2))+SIGN(DELAY(CLOSE,2)-DELAY(CLOSE,3)))*SUM(VOLUME,5)/SUM(VOLUME,20)
        fields.append(
            "-1*Rank(Sign($close-Ref($close,1))+Sign(Ref($close,1)-Ref($close,2))"
            "+Sign(Ref($close,2)-Ref($close,3)),1)*Sum($volume,5)/Sum($volume,20)"
        )
        names.append("GTJA048")

        # 049: SUM((HIGH+LOW>=REF(HIGH,1)+REF(LOW,1))?0:MAX(ABS(HIGH-REF(HIGH,1)),ABS(LOW-REF(LOW,1))),12)/(SUM((HIGH+LOW>=REF(HIGH,1)+REF(LOW,1))?0:MAX(ABS(HIGH-REF(HIGH,1)),ABS(LOW-REF(LOW,1))),12)+SUM((HIGH+LOW<=REF(HIGH,1)+REF(LOW,1))?0:MAX(ABS(HIGH-REF(HIGH,1)),ABS(LOW-REF(LOW,1))),12))
        # Aroon-like indicator
        fields.append(
            "Sum(If($high+$low>=Ref($high,1)+Ref($low,1), 0, "
            "Greater(Abs($high-Ref($high,1)),Abs($low-Ref($low,1)))), 12)"
            "/(Sum(If($high+$low>=Ref($high,1)+Ref($low,1), 0, "
            "Greater(Abs($high-Ref($high,1)),Abs($low-Ref($low,1)))), 12)"
            "+Sum(If($high+$low<=Ref($high,1)+Ref($low,1), 0, "
            "Greater(Abs($high-Ref($high,1)),Abs($low-Ref($low,1)))), 12)+1e-12)"
        )
        names.append("GTJA049")

        # 050: -1*TSMAX(RANK(CORR(RANK(VOLUME),RANK(VWAP),5)),5)
        fields.append(
            "-1*Max(Rank(Corr(Rank($volume,1),Rank($vwap,1),5),1),5)"
        )
        names.append("GTJA050")

        # 051: same formula structure as #049 with SUM of conditional max
        # Simplified
        fields.append(
            "Sum(If($high+$low>=Ref($high,1)+Ref($low,1), 0, "
            "Greater(Abs($high-Ref($high,1)),Abs($low-Ref($low,1)))), 12)"
            "/(Sum(If($high+$low>=Ref($high,1)+Ref($low,1), 0, "
            "Greater(Abs($high-Ref($high,1)),Abs($low-Ref($low,1)))), 12)"
            "+Sum(If($high+$low<=Ref($high,1)+Ref($low,1), 0, "
            "Greater(Abs($high-Ref($high,1)),Abs($low-Ref($low,1)))), 12)+1e-12)"
        )
        names.append("GTJA051")

        # 052: SUM(MAX(0,HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)),26)/SUM(MAX(0,DELAY((HIGH+LOW+CLOSE)/3,1)-LOW),26)*100
        tp = "($high+$low+$close)/3"
        fields.append(
            f"Sum(Greater($high-Ref({tp},1),0),26)"
            f"/(Sum(Greater(Ref({tp},1)-$low,0),26)+1e-12)*100"
        )
        names.append("GTJA052")

        # 053: COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100
        fields.append(
            "Mean($close>Ref($close,1), 12)*100"
        )
        names.append("GTJA053")

        # 054: -1*RANK(STD(ABS(CLOSE-OPEN))+CLOSE-OPEN+CORR(CLOSE,OPEN,10))
        fields.append(
            "-1*Rank(Std(Abs($close-$open),1)+($close-$open)+Corr($close,$open,10), 1)"
        )
        names.append("GTJA054")

        # 055: SUM(16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/((ABS(HIGH-DELAY(CLOSE,1))>ABS(LOW-DELAY(CLOSE,1))&ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)))?ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))&ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1)))?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4)*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1))),20)
        # This is a very complex formula - simplified approximation
        fields.append(
            "Sum(($close/Ref($close,1)-1)*$volume/Mean($volume,20), 20)"
        )
        names.append("GTJA055")

        # 056: -1*RANK(SUM(RETURNS,10)/SUM(SUM(RETURNS,2),3))*RANK(RETURNS*CAP)
        # [CAP not available, simplified]
        fields.append(
            "-1*Rank(Sum($close/Ref($close,1)-1,10)/Sum(Sum($close/Ref($close,1)-1,2),3),1)"
            "*Rank($close/Ref($close,1)-1,1)"
        )
        names.append("GTJA056")

        # 057: SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100, 3, 1)
        fields.append(
            "SMA(($close-Min($low,9))/(Max($high,9)-Min($low,9)+1e-12)*100, 3, 1)"
        )
        names.append("GTJA057")

        # 058: COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100
        fields.append(
            "Mean($close>Ref($close,1), 20)*100"
        )
        names.append("GTJA058")

        # 059: SUM(CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0),20)
        fields.append(
            "Sum(If($close>Ref($close,1), $volume, If($close<Ref($close,1), -$volume, 0)), 20)"
        )
        names.append("GTJA059")

        # 060: SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME,20)
        fields.append(
            "Sum((($close-$low)-($high-$close))/($high-$low+1e-12)*$volume, 20)"
        )
        names.append("GTJA060")

        # 061: -1*RANK(MAX(VWAP-CLOSE,12))
        fields.append(
            "-1*Rank(Max($vwap-$close,12), 1)"
        )
        names.append("GTJA061")

        # 062: -1*CORR(HIGH, RANK(VOLUME), 5)
        fields.append(
            "-1*Corr($high, Rank($volume,1), 5)"
        )
        names.append("GTJA062")

        # 063: SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100
        fields.append(
            "SMA(Greater($close-Ref($close,1),0),6,1)"
            "/(SMA(Abs($close-Ref($close,1)),6,1)+1e-12)*100"
        )
        names.append("GTJA063")

        # 064: SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100
        fields.append(
            "SMA(Greater($close-Ref($close,1),0),12,1)"
            "/(SMA(Abs($close-Ref($close,1)),12,1)+1e-12)*100"
        )
        names.append("GTJA064")

        # 065: MEAN(CLOSE,6)/CLOSE
        fields.append(
            "Mean($close,6)/$close"
        )
        names.append("GTJA065")

        # 066: (CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100
        fields.append(
            "($close-Mean($close,6))/Mean($close,6)*100"
        )
        names.append("GTJA066")

        # 067: SMA(MAX(CLOSE-DELAY(CLOSE,1),0),24,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),24,1)*100
        fields.append(
            "SMA(Greater($close-Ref($close,1),0),24,1)"
            "/(SMA(Abs($close-Ref($close,1)),24,1)+1e-12)*100"
        )
        names.append("GTJA067")

        # 068: SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME, 15, 2)
        # NOTE: GTJA068 in original is SMA((H+L+C)/3*VOL, 13, 1), but the canonical
        # GTJA068 is SMA(midpoint_change * range / volume, 15, 2) (like #009 but n=15).
        # Using the canonical version.
        fields.append(
            "SMA((($high+$low)/2-(Ref($high,1)+Ref($low,1))/2)*($high-$low)/($volume+1e-12), 15, 2)"
        )
        names.append("GTJA068")

        # 069: DTM/DBM complex formula: SUM(DTM,20)>SUM(DBM,20)?(SD-SB)/SD:(SD=SB?0:(SD-SB)/SB)
        # DTM = IF(OPEN<=REF(OPEN,1), 0, MAX(HIGH-OPEN, OPEN-REF(OPEN,1)))
        # DBM = IF(OPEN>=REF(OPEN,1), 0, MAX(OPEN-LOW, OPEN-REF(OPEN,1)))
        dtm = "If($open<=Ref($open,1),0,Greater($high-$open,$open-Ref($open,1)))"
        dbm = "If($open>=Ref($open,1),0,Greater($open-$low,$open-Ref($open,1)))"
        sd = f"Sum({dtm},20)"
        sb = f"Sum({dbm},20)"
        fields.append(
            f"If({sd}>{sb},({sd}-{sb})/{sd},If({sd}={sb},0,({sd}-{sb})/{sb}))"
        )
        names.append("GTJA069")

        # 070: STD(AMOUNT,6)
        # Uses Amount() custom operator for exact 成交额 access
        fields.append(
            "Std(Amount(), 6)"
        )
        names.append("GTJA070")

        # 071: (CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100
        fields.append(
            "($close-Mean($close,24))/Mean($close,24)*100"
        )
        names.append("GTJA071")

        # 072: SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100, 15, 1)
        fields.append(
            "WMA((Max($high,6)-$close)/(Max($high,6)-Min($low,6)+1e-12)*100, 15)"
        )
        names.append("GTJA072")

        # 073: -1*TSRANK(DECAYLINEAR(CORR(CLOSE,VOLUME,10),16),4)
        fields.append(
            "-1*Rank(WMA(Corr($close,$volume,10),16),4)"
        )
        names.append("GTJA073")

        # 074: RANK(CORR(SUM(LOW*0.35+VWAP*0.65,20),SUM(MEAN(VOLUME,60),20),7))
        fields.append(
            "Rank(Corr(Sum($low*0.35+$vwap*0.65,20),Sum(Mean($volume,60),20),7),1)"
        )
        names.append("GTJA074")

        # 075: COUNT(CLOSE>OPEN & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)/COUNT(BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)
        # [BENCHMARK data not available, simplified to close>open count ratio]
        fields.append(
            "Mean($close>$open, 50)"
        )
        names.append("GTJA075")

        # 076: STD(ABS(CLOSE/DELAY(CLOSE,1)-1)/VOLUME,20)/MEAN(ABS(CLOSE/DELAY(CLOSE,1)-1)/VOLUME,20)
        fields.append(
            "Std(Abs($close/Ref($close,1)-1)/($volume+1e-12),20)"
            "/(Mean(Abs($close/Ref($close,1)-1)/($volume+1e-12),20)+1e-12)"
        )
        names.append("GTJA076")

        # 077: MIN(RANK(DECAYLINEAR((HIGH+LOW)/2+HIGH-(VWAP+HIGH),20)),RANK(DECAYLINEAR(CORR((HIGH+LOW)/2,MEAN(VOLUME,40),3),6)))
        fields.append(
            "Less(Rank(WMA(($high+$low)/2+$high-($vwap+$high),20),1),"
            "Rank(WMA(Corr(($high+$low)/2,Mean($volume,40),3),6),1))"
        )
        names.append("GTJA077")

        # 078: (HIGH+LOW+CLOSE)/3*VOLUME
        fields.append(
            "($high+$low+$close)/3*$volume"
        )
        names.append("GTJA078")

        # 079: SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100
        fields.append(
            "SMA(Greater($close-Ref($close,1),0),12,1)"
            "/(SMA(Abs($close-Ref($close,1)),12,1)+1e-12)*100"
        )
        names.append("GTJA079")

        # 080: (VOLUME-DELAY(VOLUME,1))/DELAY(VOLUME,1)*100
        fields.append(
            "($volume-Ref($volume,1))/Ref($volume,1)*100"
        )
        names.append("GTJA080")

        # 081: SMA(VOLUME, 10, 1)
        fields.append(
            "SMA($volume, 10, 1)"
        )
        names.append("GTJA081")

        # 082: SMA(HIGH, 10, 1)/SMA(LOW, 10, 1)*100
        fields.append(
            "SMA($high,10,1)/(SMA($low,10,1)+1e-12)*100"
        )
        names.append("GTJA082")

        # 083: -1*RANK(COVIANCE(RANK(HIGH),RANK(VOLUME),5))
        fields.append(
            "-1*Rank(Cov(Rank($high,1),Rank($volume,1),5),1)"
        )
        names.append("GTJA083")

        # 084: SUM(CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0),20)
        fields.append(
            "Sum(If($close>Ref($close,1), $volume, If($close<Ref($close,1), -$volume, 0)), 20)"
        )
        names.append("GTJA084")

        # 085: TSRANK(VOLUME/MEAN(VOLUME,20),20)*TSRANK(-1*DELTA(CLOSE,7),8)
        fields.append(
            "Rank($volume/Mean($volume,20),20)*Rank(-1*Delta($close,7),8)"
        )
        names.append("GTJA085")

        # 086: (DELAY(CLOSE,20)-DELAY(CLOSE,10))/10-(DELAY(CLOSE,10)-CLOSE)/10
        fields.append(
            "(Ref($close,20)-Ref($close,10))/10-(Ref($close,10)-$close)/10"
        )
        names.append("GTJA086")

        # 087: -1*RANK(DECAYLINEAR(DELTA(VWAP,3.51013),7.23052))
        fields.append(
            "-1*Rank(WMA(Delta($vwap,4),7),1)"
        )
        names.append("GTJA087")

        # 088: (CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100
        fields.append(
            "($close-Ref($close,20))/Ref($close,20)*100"
        )
        names.append("GTJA088")

        # 089: 2*SMA(CLOSE,13,2)-SMA(CLOSE,27,2)
        fields.append(
            "2*SMA($close,13,2)-SMA($close,27,2)"
        )
        names.append("GTJA089")

        # 090: (CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5)*100
        fields.append(
            "($close-Ref($close,5))/Ref($close,5)*100"
        )
        names.append("GTJA090")

        # 091: -1*RANK(DECAYLINEAR(DELTA(CLOSE,2.25164),8.22237))
        fields.append(
            "-1*Rank(WMA(Delta($close,2),8),1)"
        )
        names.append("GTJA091")

        # 092: -1*RANK(DECAYLINEAR(CORR(HIGH,VOLUME,5.11425),4.51099))
        fields.append(
            "-1*Rank(WMA(Corr($high,$volume,5),5),1)"
        )
        names.append("GTJA092")

        # 093: -1*TSRANK(DECAYLINEAR(CORR(RANK(VWAP),RANK(VOLUME),5.07876),14.4539),4.41347)
        fields.append(
            "-1*Rank(WMA(Corr(Rank($vwap,1),Rank($volume,1),5),14),4)"
        )
        names.append("GTJA093")

        # 094: -1*RANK(DECAYLINEAR(CORR(CLOSE,MEAN(VOLUME,60),9.17385),14.3857))
        fields.append(
            "-1*Rank(WMA(Corr($close,Mean($volume,60),9),14),1)"
        )
        names.append("GTJA094")

        # 095: -1*RANK(DECAYLINEAR(CORR(RANK(HIGH),RANK(MEAN(VOLUME,15)),8.91965),7.67903))
        fields.append(
            "-1*Rank(WMA(Corr(Rank($high,1),Rank(Mean($volume,15),1),9),8),1)"
        )
        names.append("GTJA095")

        # 096: -1*TSMAX(RANK(CORR(RANK(VWAP),RANK(VOLUME),5)),5)
        fields.append(
            "-1*Max(Rank(Corr(Rank($vwap,1),Rank($volume,1),5),1),5)"
        )
        names.append("GTJA096")

        # 097: STD(VOLUME,10)
        fields.append(
            "Std($volume, 10)"
        )
        names.append("GTJA097")

        # 098: ((DELTA(SUM(CLOSE,100)/100,100)/DELAY(CLOSE,100))<=0.05)?-1*(CLOSE-TSMIN(CLOSE,100)):-1*DELTA(CLOSE,3)
        fields.append(
            "If(Delta(Mean($close,100),100)/Ref($close,100)<=0.05, "
            "-1*($close-Min($close,100)), "
            "-1*Delta($close,3))"
        )
        names.append("GTJA098")

        # 099: -1*RANK(COVIANCE(RANK(CLOSE),RANK(VOLUME),5))
        fields.append(
            "-1*Rank(Cov(Rank($close,1),Rank($volume,1),5),1)"
        )
        names.append("GTJA099")

        # 100: STD(VOLUME,20)
        fields.append(
            "Std($volume, 20)"
        )
        names.append("GTJA100")

        # 101: (CLOSE-OPEN)/(HIGH-LOW+1e-12)*VOLUME
        fields.append(
            "($close-$open)/($high-$low+1e-12)*$volume"
        )
        names.append("GTJA101")

        # 102: SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100
        fields.append(
            "SMA(Greater($close-Ref($close,1),0),6,1)"
            "/(SMA(Abs($close-Ref($close,1)),6,1)+1e-12)*100"
        )
        names.append("GTJA102")

        # 103: (20-(HIGH-LOW)/STD(CLOSE,20))*100
        fields.append(
            "(20-($high-$low)/(Std($close,20)+1e-12))*100"
        )
        names.append("GTJA103")

        # 104: -1*DELTA(CORR(HIGH,VOLUME,5),5)*RANK(STD(CLOSE,20))
        fields.append(
            "-1*Delta(Corr($high,$volume,5),5)*Rank(Std($close,20),1)"
        )
        names.append("GTJA104")

        # 105: -1*CORR(RANK(OPEN),RANK(VOLUME),10)
        fields.append(
            "-1*Corr(Rank($open,1),Rank($volume,1),10)"
        )
        names.append("GTJA105")

        # 106: CLOSE-DELAY(CLOSE,20)
        fields.append(
            "Delta($close, 20)"
        )
        names.append("GTJA106")

        # 107: ((-1*RANK(DELTA(OPEN,1)))*RANK(OPEN-DELAY(CLOSE,1)))*RANK(DELTA(VOLUME,1))
        fields.append(
            "(-1*Rank(Delta($open,1),1))*Rank($open-Ref($close,1),1)*Rank(Delta($volume,1),1)"
        )
        names.append("GTJA107")

        # 108: RANK(HIGH-MIN(HIGH,2))^RANK(CORR(VWAP,MEAN(VOLUME,120),6))
        # Approximate power operation
        fields.append(
            "Sign(Rank($high-Min($high,2),1))*Power(Abs(Rank($high-Min($high,2),1)), "
            "Abs(Rank(Corr($vwap,Mean($volume,120),6),1)))"
        )
        names.append("GTJA108")

        # 109: SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)
        fields.append(
            "SMA($high-$low,10,2)/SMA(SMA($high-$low,10,2),10,2)"
        )
        names.append("GTJA109")

        # 110: SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100
        fields.append(
            "Sum(Greater($high-Ref($close,1),0),20)"
            "/(Sum(Greater(Ref($close,1)-$low,0),20)+1e-12)*100"
        )
        names.append("GTJA110")

        # 111: SMA(VOLUME*(CLOSE-LOW-(HIGH-CLOSE))/(HIGH-LOW+1e-12),11,2)-SMA(VOLUME*(CLOSE-LOW-(HIGH-CLOSE))/(HIGH-LOW+1e-12),4,2)
        mfm = "$volume*(($close-$low)-($high-$close))/($high-$low+1e-12)"
        fields.append(
            f"SMA({mfm},11,2)-SMA({mfm},4,2)"
        )
        names.append("GTJA111")

        # 112: (SUM(CLOSE>DELAY(CLOSE,1)?VOLUME:0,12)-SUM(CLOSE<DELAY(CLOSE,1)?VOLUME:0,12))/(SUM(CLOSE>DELAY(CLOSE,1)?VOLUME:0,12)+SUM(CLOSE<DELAY(CLOSE,1)?VOLUME:0,12))*100
        fields.append(
            "(Sum(If($close>Ref($close,1),$volume,0),12)-Sum(If($close<Ref($close,1),$volume,0),12))"
            "/(Sum(If($close>Ref($close,1),$volume,0),12)+Sum(If($close<Ref($close,1),$volume,0),12)+1e-12)*100"
        )
        names.append("GTJA112")

        # 113: -1*RANK(SUM(DELAY(CLOSE,5),20)/20)*CORR(CLOSE,VOLUME,2)*RANK(CORR(SUM(CLOSE,5),SUM(CLOSE,20),2))
        fields.append(
            "-1*Rank(Sum(Ref($close,5),20)/20,1)*Corr($close,$volume,2)"
            "*Rank(Corr(Sum($close,5),Sum($close,20),2),1)"
        )
        names.append("GTJA113")

        # 114: RANK(DELAY((RETURN<0?STD(RETURN,20):CLOSE)^2,5))
        fields.append(
            "Rank(Ref(Power(If($close/Ref($close,1)-1<0, "
            "Std($close/Ref($close,1)-1,20), $close), 2), 5), 1)"
        )
        names.append("GTJA114")

        # 115: -1*RANK(CORR(HIGH,VOLUME,30))*RANK(HIGH)
        fields.append(
            "-1*Rank(Corr($high,$volume,30),1)*Rank($high,1)"
        )
        names.append("GTJA115")

        # 116: REGBETA(CLOSE,SEQUENCE,20)
        fields.append(
            "Slope($close, 20)/$close"
        )
        names.append("GTJA116")

        # 117: TSRANK(VOLUME,32)*(1-TSRANK(CLOSE+HIGH-LOW,16))*(1-TSRANK(RETURN,32))
        fields.append(
            "Rank($volume,32)*(1-Rank($close+$high-$low,16))*(1-Rank($close/Ref($close,1)-1,32))"
        )
        names.append("GTJA117")

        # 118: SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100
        fields.append(
            "Sum($high-$open,20)/(Sum($open-$low,20)+1e-12)*100"
        )
        names.append("GTJA118")

        # 119: RANK(DECAYLINEAR(CORR(VWAP,SUM(MEAN(VOLUME,5),26.4719),4.58418),7.18088))-RANK(DECAYLINEAR(TSRANK(TSARGMIN(CORR(RANK(OPEN),RANK(MEAN(VOLUME,15)),20.8187),8.62571),6.95668),8.07206))
        # Exact with custom operator TsArgmin
        fields.append(
            "Rank(WMA(Corr($vwap,Sum(Mean($volume,5),26),5),7),1)"
            "-Rank(WMA(Rank(TsArgmin(Corr(Rank($open,1),Rank(Mean($volume,15),1),21),9),7),8),1)"
        )
        names.append("GTJA119")

        # 120: RANK(VWAP-CLOSE)/RANK(VWAP+CLOSE)
        fields.append(
            "Rank($vwap-$close,1)/Rank($vwap+$close,1)"
        )
        names.append("GTJA120")

        # 121: RANK(VWAP-MIN(VWAP,12))^TSRANK(CORR(TSRANK(VWAP,20),TSRANK(MEAN(VOLUME,60),2),18),3)
        fields.append(
            "Sign(Rank($vwap-Min($vwap,12),1))*Power(Abs(Rank($vwap-Min($vwap,12),1)), "
            "Abs(Rank(Corr(Rank($vwap,20),Rank(Mean($volume,60),2),18),3)))"
        )
        names.append("GTJA121")

        # 122: SMA(SMA(LOG(CLOSE),13,2)-SMA(LOG(CLOSE),27,2),2,1)
        fields.append(
            "SMA(SMA(Log($close),13,2)-SMA(Log($close),27,2),2,1)"
        )
        names.append("GTJA122")

        # 123: RANK(CORR(SUM((HIGH+LOW)/2,20),SUM(MEAN(VOLUME,60),20),9))*RANK(CORR(LOW,VOLUME,6))
        fields.append(
            "Rank(Corr(Sum(($high+$low)/2,20),Sum(Mean($volume,60),20),9),1)"
            "*Rank(Corr($low,$volume,6),1)"
        )
        names.append("GTJA123")

        # 124: (CLOSE-VWAP)/DECAYLINEAR(RANK(TSARGMAX(CLOSE,30)),2)
        # Exact with custom operator TsArgmax
        fields.append(
            "($close-$vwap)/WMA(Rank(TsArgmax($close,30),1),2)"
        )
        names.append("GTJA124")

        # 125: RANK(DECAYLINEAR(CORR(VWAP,MEAN(VOLUME,17),4.81854),6.49929))*RANK(DECAYLINEAR(TSRANK(CORR(RANK(LOW),RANK(MEAN(VOLUME,10)),4.81854),6.49929),2.02853))
        fields.append(
            "Rank(WMA(Corr($vwap,Mean($volume,17),5),6),1)"
            "*Rank(WMA(Rank(Corr(Rank($low,1),Rank(Mean($volume,10),1),5),6),2),1)"
        )
        names.append("GTJA125")

        # 126: (CLOSE+HIGH+LOW)/3
        fields.append(
            "($close+$high+$low)/3"
        )
        names.append("GTJA126")

        # 127: (MEAN((100*(CLOSE-MAX(CLOSE,12))/(MAX(CLOSE,12)+1e-12))^2))^(1/2)
        fields.append(
            "Power(Power(100*($close-Max($close,12))/(Max($close,12)+1e-12), 2), 0.5)"
        )
        names.append("GTJA127")

        # 128: 100-(100/(1+SUM((HIGH+LOW+CLOSE>REF(HIGH,1)+REF(LOW,1)+REF(CLOSE,1)?MAX(HIGH,REF(CLOSE,1)):MIN(LOW,REF(CLOSE,1))),14)))
        # Simplified
        tp3 = "($high+$low+$close)"
        fields.append(
            f"100-(100/(1+Sum(If({tp3}>Ref({tp3},1),Greater($high,Ref($close,1)),Less($low,Ref($close,1))),14)))"
        )
        names.append("GTJA128")

        # 129: SUM((CLOSE-DELAY(CLOSE,1)<0)?ABS(CLOSE-DELAY(CLOSE,1)):0,12)
        fields.append(
            "Sum(If($close<Ref($close,1), Abs($close-Ref($close,1)), 0), 12)"
        )
        names.append("GTJA129")

        # 130: RANK(DECAYLINEAR(CORR((HIGH+LOW)/2,MEAN(VOLUME,40),9),10))/RANK(DECAYLINEAR(CORR(RANK(VWAP),RANK(VOLUME),7),3))
        fields.append(
            "Rank(WMA(Corr(($high+$low)/2,Mean($volume,40),9),10),1)"
            "/Rank(WMA(Corr(Rank($vwap,1),Rank($volume,1),7),3),1)"
        )
        names.append("GTJA130")

        # 131: RANK(DELTA(VWAP,1))^TSRANK(CORR(CLOSE,MEAN(VOLUME,50),18),18)
        fields.append(
            "Sign(Rank(Delta($vwap,1),1))*Power(Abs(Rank(Delta($vwap,1),1)), "
            "Abs(Rank(Corr($close,Mean($volume,50),18),18)))"
        )
        names.append("GTJA131")

        # 132: MEAN(AMOUNT,20)
        fields.append(
            "Mean(Amount(), 20)"
        )
        names.append("GTJA132")

        # 133: (20-(HIGH-LOW)/STD(CLOSE,20))*100
        fields.append(
            "(20-($high-$low)/(Std($close,20)+1e-12))*100"
        )
        names.append("GTJA133")

        # 134: (CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)*VOLUME
        fields.append(
            "($close-Ref($close,12))/Ref($close,12)*$volume"
        )
        names.append("GTJA134")

        # 135: SMA(DELAY(CLOSE/DELAY(CLOSE,20),1),20,1)
        fields.append(
            "SMA(Ref($close/Ref($close,20),1),20,1)"
        )
        names.append("GTJA135")

        # 136: -1*RANK(DELTA(CLOSE/DELAY(CLOSE,1),3))*CORR(OPEN,VOLUME,10)
        fields.append(
            "-1*Rank(Delta($close/Ref($close,1),3),1)*Corr($open,$volume,10)"
        )
        names.append("GTJA136")

        # 137: -1*RANK(DECAYLINEAR(DELTA(CLOSE,2),8))*RANK(DECAYLINEAR(CORR(VWAP,MEAN(VOLUME,20),8),6))
        fields.append(
            "-1*Rank(WMA(Delta($close,2),8),1)*Rank(WMA(Corr($vwap,Mean($volume,20),8),6),1)"
        )
        names.append("GTJA137")

        # 138: -1*RANK(DECAYLINEAR(STD(LOW,10),6))*RANK(DECAYLINEAR(CORR(LOW,MEAN(VOLUME,10),10),6))
        fields.append(
            "-1*Rank(WMA(Std($low,10),6),1)*Rank(WMA(Corr($low,Mean($volume,10),10),6),1)"
        )
        names.append("GTJA138")

        # 139: -1*RANK(DELTA(CLOSE,3))*CORR(OPEN,VOLUME,10)
        fields.append(
            "-1*Rank(Delta($close,3),1)*Corr($open,$volume,10)"
        )
        names.append("GTJA139")

        # 140: MIN(RANK(DECAYLINEAR(RANK(OPEN)+RANK(LOW)-RANK(HIGH)-RANK(CLOSE),8)),TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE,8),TSRANK(MEAN(VOLUME,60),20),8),7),3))
        fields.append(
            "Less(Rank(WMA(Rank($open,1)+Rank($low,1)-Rank($high,1)-Rank($close,1),8),1),"
            "Rank(WMA(Corr(Rank($close,8),Rank(Mean($volume,60),20),8),7),3))"
        )
        names.append("GTJA140")

        # 141: RANK(CORR(RANK(HIGH),RANK(MEAN(VOLUME,15)),9))*(-1)
        fields.append(
            "-1*Rank(Corr(Rank($high,1),Rank(Mean($volume,15),1),9),1)"
        )
        names.append("GTJA141")

        # 142: -1*RANK(TSRANK(CLOSE,10))*RANK(DELTA(DELTA(CLOSE,1),1))*RANK(TSRANK(VOLUME/MEAN(VOLUME,20),5))
        fields.append(
            "-1*Rank(Rank($close,10),1)*Rank(Delta(Delta($close,1),1),1)"
            "*Rank(Rank($volume/Mean($volume,20),5),1)"
        )
        names.append("GTJA142")

        # 143: CLOSE>DELAY(CLOSE,1)?(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1):SELF/MEAN(CLOSE,3)
        # [SELF reference not available in Qlib expression]
        fields.append(
            "If($close>Ref($close,1), ($close-Ref($close,1))/Ref($close,1), 1/Mean($close,3))"
        )
        names.append("GTJA143")

        # 144: SUMIF(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT,20,CLOSE<DELAY(CLOSE,1))/COUNT(CLOSE<DELAY(CLOSE,1),20)
        fields.append(
            "Sum(If($close<Ref($close,1), Abs($close/Ref($close,1)-1)/(Amount()+1e-12), 0), 20)"
            "/(Sum(If($close<Ref($close,1), 1, 0), 20)+1e-12)"
        )
        names.append("GTJA144")

        # 145: (MEAN(VOLUME,9)-MEAN(VOLUME,26))/MEAN(VOLUME,12)*100
        fields.append(
            "(Mean($volume,9)-Mean($volume,26))/Mean($volume,12)*100"
        )
        names.append("GTJA145")

        # 146: MEAN((CLOSE-DELAY(CLOSE,1)-(MEAN(CLOSE,20)-MEAN(CLOSE,20).shift(1)))/DELAY(CLOSE,1)-(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),60)
        fields.append(
            "Mean(($close/Ref($close,1)-1-(Mean($close,20)-Ref(Mean($close,20),1))/Ref($close,1)), 60)"
        )
        names.append("GTJA146")

        # 147: REGBETA(MEAN(CLOSE,12),SEQUENCE(12))
        fields.append(
            "Slope(Mean($close,12), 12)/$close"
        )
        names.append("GTJA147")

        # 148: RANK(CORR(OPEN,SUM(MEAN(VOLUME,60),9),9))*RANK(OPEN-CLOSE+OPEN-DELAY(CLOSE,1))
        fields.append(
            "Rank(Corr($open,Sum(Mean($volume,60),9),9),1)"
            "*Rank($open-$close+$open-Ref($close,1),1)"
        )
        names.append("GTJA148")

        # 149: REGBETA(MEAN(VOLUME,12),SEQUENCE(12))
        fields.append(
            "Slope(Mean($volume,12), 12)/($volume+1e-12)"
        )
        names.append("GTJA149")

        # 150: (CLOSE+HIGH+LOW)/3*VOLUME
        fields.append(
            "($close+$high+$low)/3*$volume"
        )
        names.append("GTJA150")

        # 151: SMA(CLOSE-DELAY(CLOSE,20),20,1)
        fields.append(
            "SMA(Delta($close,20), 20, 1)"
        )
        names.append("GTJA151")

        # 152: SMA(MEAN(DELAY(SMA(HIGH-LOW,9,1),1),12,1)-MEAN(DELAY(SMA(HIGH-LOW,9,1),1),26,1),9,1)
        fields.append(
            "SMA(Mean(Ref(SMA($high-$low,9,1),1),12)-Mean(Ref(SMA($high-$low,9,1),1),26),9,1)"
        )
        names.append("GTJA152")

        # 153: (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)
        fields.append(
            "(Mean($close,3)+Mean($close,6)+Mean($close,12)+Mean($close,24))/(4*$close)"
        )
        names.append("GTJA153")

        # 154: (VWAP-MIN(VWAP,16))<CORR(VWAP,MEAN(VOLUME,180),18)
        fields.append(
            "($vwap-Min($vwap,16))-Corr($vwap,Mean($volume,180),18)"
        )
        names.append("GTJA154")

        # 155: SMA(VOLUME,13,2)-SMA(VOLUME,27,2)
        fields.append(
            "SMA($volume,13,2)-SMA($volume,27,2)"
        )
        names.append("GTJA155")

        # 156: MAX(RANK(DECAYLINEAR(DELTA(VWAP,5),3)),RANK(DECAYLINEAR((DELTA(OPEN*0.85+HIGH*0.15,2)/OPEN*0.85+HIGH*0.15)*100,3)))*(-1)
        fields.append(
            "-1*Greater(Rank(WMA(Delta($vwap,5),3),1),"
            "Rank(WMA(Delta($open*0.85+$high*0.15,2)/($open*0.85+$high*0.15)*100,3),1))"
        )
        names.append("GTJA156")

        # 157: MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK(-1*RANK(DELTA(CLOSE-OPEN,5))))),2),1)))),1),5)+TSRANK(DELTA(VWAP,1),5)
        # Simplified
        fields.append(
            "Less(Rank(Rank(Log(Sum(Min(Rank(Rank(-1*Rank(Delta($close-$open,5),1)),1),1),2),1)),1),1),5)"
            "+Rank(Delta($vwap,1),5)"
        )
        names.append("GTJA157")

        # 158: (HIGH-LOW)/CLOSE
        fields.append(
            "($high-$low)/$close"
        )
        names.append("GTJA158")

        # 159: -1*RANK(CLOSE)*RANK(DELTA(CLOSE,1))
        fields.append(
            "-1*Rank($close,1)*Rank(Delta($close,1),1)"
        )
        names.append("GTJA159")

        # 160: MIN(PROD(RANK(RANK(DECAYLINEAR(-1*RANK(DELTA(CLOSE,2)),8))),1),5)+TSRANK(DELTA(VWAP,1),5)
        fields.append(
            "Less(Rank(Rank(WMA(-1*Rank(Delta($close,2),1),8),1),1),5)+Rank(Delta($vwap,1),5)"
        )
        names.append("GTJA160")

        # 161: MEAN(MAX(MAX(HIGH-LOW,ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)
        fields.append(
            "Mean(Greater(Greater($high-$low,Abs(Ref($close,1)-$high)),Abs(Ref($close,1)-$low)),12)"
        )
        names.append("GTJA161")

        # 162: RSI stochastic (SMA-based)
        rsi = "SMA(Greater($close-Ref($close,1),0),12,1)/(SMA(Abs($close-Ref($close,1)),12,1)+1e-12)*100"
        fields.append(
            f"({rsi}-Min({rsi},12))/(Max({rsi},12)-Min({rsi},12)+1e-12)"
        )
        names.append("GTJA162")

        # 163: RANK((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0)^2)
        fields.append(
            "Rank(Power(If($close>Ref($close,1), Std($close,20), 0), 2), 1)"
        )
        names.append("GTJA163")

        # 164: SMA((CLOSE>DELAY(CLOSE,1)?1/(CLOSE-DELAY(CLOSE,1)):1-MIN(1,ABS(CLOSE-DELAY(CLOSE,1)))),12,1)/(HIGH-LOW+1e-12)*100
        fields.append(
            "SMA(If($close>Ref($close,1), 1/($close-Ref($close,1)+1e-12), "
            "1-Less(1,Abs($close-Ref($close,1)))), 12, 1)/($high-$low+1e-12)*100"
        )
        names.append("GTJA164")

        # 165: MAX(SUMAC(CLOSE,MEAN(VOLUME,20)))/MIN(SUMAC(CLOSE,MEAN(VOLUME,20)))
        # Simplified: close / MA20 ratio range
        fields.append(
            "Max($close/Mean($volume,20), 60)/(Min($close/Mean($volume,20), 60)+1e-12)"
        )
        names.append("GTJA165")

        # 166: -20*(20-1)^1.5*SUM(CLOSE/DELAY(CLOSE,1)-1-MEAN(CLOSE/DELAY(CLOSE,1)-1,20),20)/((20-1)*(20-2)*(SUM((CLOSE/DELAY(CLOSE,1)-1)^2,20))^1.5)
        # Complex skewness-like formula, simplified
        fields.append(
            "-20*Power(19,1.5)*Sum($close/Ref($close,1)-1-Mean($close/Ref($close,1)-1,20),20)"
            "/(19*18*Power(Sum(Power($close/Ref($close,1)-1,2),20),1.5)+1e-12)"
        )
        names.append("GTJA166")

        # 167: SUM(CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0,12)
        fields.append(
            "Sum(Greater($close-Ref($close,1),0), 12)"
        )
        names.append("GTJA167")

        # 168: -1*VOLUME/MEAN(VOLUME,20)
        fields.append(
            "-1*$volume/Mean($volume,20)"
        )
        names.append("GTJA168")

        # 169: SMA(MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),12)-MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),26),9,1)
        fields.append(
            "SMA(Mean(Ref(SMA($close-Ref($close,1),9,1),1),12)-Mean(Ref(SMA($close-Ref($close,1),9,1),1),26),9,1)"
        )
        names.append("GTJA169")

        # 170: RANK((CLOSE/DELAY(CLOSE,1)-1)*VOLUME)*TSRANK(VWAP-MAX(VWAP,12),16)
        fields.append(
            "Rank(($close/Ref($close,1)-1)*$volume,1)*Rank($vwap-Max($vwap,12),16)"
        )
        names.append("GTJA170")

        # 171: -1*RANK(LOW)*RANK(OPEN)*RANK(HIGH)*RANK(CLOSE)
        fields.append(
            "-1*Rank($low,1)*Rank($open,1)*Rank($high,1)*Rank($close,1)"
        )
        names.append("GTJA171")

        # 172: MEAN(ABS(SUM(LN(CLOSE/DELAY(CLOSE,1)),6)/6-(LN(CLOSE/DELAY(CLOSE,20))/20)),15)
        fields.append(
            "Mean(Abs(Sum(Log($close/Ref($close,1)),6)/6-Log($close/Ref($close,20))/20),15)"
        )
        names.append("GTJA172")

        # 173: 3*SMA(CLOSE,13,2)-2*SMA(SMA(CLOSE,13,2),3,1)
        fields.append(
            "3*SMA($close,13,2)-2*SMA(SMA($close,13,2),3,1)"
        )
        names.append("GTJA173")

        # 174: SMA(CLOSE<DELAY(CLOSE,1)?STD(CLOSE,20):0,20,1)
        fields.append(
            "SMA(If($close<Ref($close,1), Std($close,20), 0), 20, 1)"
        )
        names.append("GTJA174")

        # 175: MEAN(MAX(MAX(HIGH-LOW,ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),6)
        fields.append(
            "Mean(Greater(Greater($high-$low,Abs(Ref($close,1)-$high)),Abs(Ref($close,1)-$low)),6)"
        )
        names.append("GTJA175")

        # 176: CORR(RANK((CLOSE-TSMIN(LOW,12))/(TSMAX(HIGH,12)-TSMIN(LOW,12))),RANK(VOLUME),6)
        fields.append(
            "Corr(Rank(($close-Min($low,12))/(Max($high,12)-Min($low,12)+1e-12),1),Rank($volume,1),6)"
        )
        names.append("GTJA176")

        # 177: (20-(HIGH-LOW)/STD(CLOSE,20))*100
        fields.append(
            "(20-($high-$low)/(Std($close,20)+1e-12))*100"
        )
        names.append("GTJA177")

        # 178: (CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*VOLUME
        fields.append(
            "($close-Ref($close,1))/Ref($close,1)*$volume"
        )
        names.append("GTJA178")

        # 179: RANK(CORR(VWAP,VOLUME,4))*RANK(CORR(RANK(LOW),RANK(MEAN(VOLUME,50)),12))
        fields.append(
            "Rank(Corr($vwap,$volume,4),1)*Rank(Corr(Rank($low,1),Rank(Mean($volume,50),1),12),1)"
        )
        names.append("GTJA179")

        # 180: MEAN(VOLUME,7)/MEAN(VOLUME,20)
        fields.append(
            "Mean($volume,7)/Mean($volume,20)"
        )
        names.append("GTJA180")

        # 181: SUM(CLOSE>DELAY(CLOSE,1)?VOLUME:0,20)/MEAN(VOLUME,20)
        fields.append(
            "Sum(If($close>Ref($close,1), $volume, 0), 20)/Mean($volume,20)"
        )
        names.append("GTJA181")

        # 182: COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100
        fields.append(
            "Mean($close>Ref($close,1), 20)*100"
        )
        names.append("GTJA182")

        # 183: COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100
        fields.append(
            "Mean($close>Ref($close,1), 20)*100"
        )
        names.append("GTJA183")

        # 184: RANK(CORR(DELAY(OPEN-CLOSE,1),CLOSE,200))+RANK(OPEN-CLOSE)
        fields.append(
            "Rank(Corr(Ref($open-$close,1),$close,200),1)+Rank($open-$close,1)"
        )
        names.append("GTJA184")

        # 185: -1*RANK(ABS(CLOSE-DELAY(CLOSE,1)))*CORR(CLOSE,VOLUME,10)
        fields.append(
            "-1*Rank(Abs($close-Ref($close,1)),1)*Corr($close,$volume,10)"
        )
        names.append("GTJA185")

        # 186: MEAN(LOW-DELAY(LOW,1),20)+MEAN(HIGH-DELAY(HIGH,1),20)
        fields.append(
            "Mean($low-Ref($low,1),20)+Mean($high-Ref($high,1),20)"
        )
        names.append("GTJA186")

        # 187: SUM(OPEN>DELAY(OPEN,1)?0:MAX(OPEN-DELAY(OPEN,1),ABS(OPEN-DELAY(OPEN,1))),20)
        fields.append(
            "Sum(If($open>Ref($open,1), 0, Greater($open-Ref($open,1),Abs($open-Ref($open,1)))), 20)"
        )
        names.append("GTJA187")

        # 188: (HIGH-LOW-SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2)*100
        fields.append(
            "($high-$low-SMA($high-$low,11,2))/(SMA($high-$low,11,2)+1e-12)*100"
        )
        names.append("GTJA188")

        # 189: MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)
        fields.append(
            "Mean(Abs($close-Mean($close,6)),6)"
        )
        names.append("GTJA189")

        # 190: LOG(COUNT(RETURN>0,20)/COUNT(RETURN<0,20))
        fields.append(
            "Log((Mean($close/Ref($close,1)-1>0,20)+1e-12)/(Mean($close/Ref($close,1)-1<0,20)+1e-12))"
        )
        names.append("GTJA190")

        # 191: CORR(MEAN(VOLUME,20),LOW,5)+((HIGH+LOW)/2)-CLOSE
        fields.append(
            "Corr(Mean($volume,20),$low,5)+(($high+$low)/2)-$close"
        )
        names.append("GTJA191")

        return fields, names
