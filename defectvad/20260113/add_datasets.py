class BaseDataset(Dataset, ABC):
    NAME = ""
    CATEGORIES = []

    # 기존 코드 생략...

    def __add__(self, other):
        """
        두 BaseDataset을 더해 새로운 BaseDataset 반환.
        결과는 두 데이터셋의 samples를 병합하고, 카테고리도 통합.
        """
        if not isinstance(other, BaseDataset):
            return NotImplemented

        # 새로운 데이터셋 생성 (기준: self 복사)
        merged = copy.copy(self)
        merged.samples = []

        # samples 병합
        merged.samples.extend(copy.deepcopy(self.samples))
        merged.samples.extend(copy.deepcopy(other.samples))

        # 카테고리 병합 및 정렬
        all_categories = set(self.category) | set(other.category)
        merged.category = sorted(all_categories)

        return merged

      def __iadd__(self, other):
        """+= 연산자 지원"""
        if not isinstance(other, BaseDataset):
            return NotImplemented
        self.samples.extend(copy.deepcopy(other.samples))
        all_categories = set(self.category) | set(other.category)
        self.category = sorted(all_categories)
        return self
